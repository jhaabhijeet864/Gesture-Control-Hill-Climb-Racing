"""
Plugin System Manager

Central coordinator for gesture plugins, mapping profiles, and actuation backends.
Handles loading, configuration, and execution of the plugin system.
"""
import os
import importlib
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

from .gesture_base import GesturePlugin, GestureResult
from .mapping_profiles import ProfileManager, MappingProfile, ActionMapping
from .actuation_backends import BackendManager, ActuationBackend
from .live_config import LiveConfigManager, ConfigSchema, create_default_schemas
from .builtin_gestures import RuleBasedGesturePlugin


log = logging.getLogger("plugin_manager")


class PluginManager:
    """Central manager for the entire plugin system."""
    
    def __init__(self, config_path: str = "config.yaml"):
        # Initialize managers
        self.config = LiveConfigManager(config_path)
        self.profile_manager = ProfileManager()
        self.backend_manager = BackendManager()
        
        # Plugin storage
        self.gesture_plugins: Dict[str, GesturePlugin] = {}
        self.active_gesture_plugins: List[str] = []
        
        # Setup default configuration schemas
        self.config.register_schemas(create_default_schemas())
        
        # Initialize system
        self._initialize_system()
        
        # Start config watching
        self.config.start_watching()
        self.config.add_watcher(self._on_config_changed)
    
    def _initialize_system(self):
        """Initialize the plugin system."""
        log.info("Initializing plugin system...")
        
        # Create default profiles if none exist
        if not self.profile_manager.list_profiles():
            self.profile_manager.create_default_profiles()
        
        # Create default backends
        backend_config = self.config.get_config("backend", {})
        created_backends = self.backend_manager.create_default_backends(backend_config)
        log.info(f"Created backends: {created_backends}")
        
        # Set active backend from config
        active_backend = self.config.get_config("backend.actuation", "pynput")
        if not self.backend_manager.set_active_backend(active_backend):
            # Fallback to first available backend
            available = self.backend_manager.list_backends()
            if available:
                self.backend_manager.set_active_backend(available[0])
                log.info(f"Using fallback backend: {available[0]}")
        
        # Load gesture plugins
        self._load_builtin_gestures()
        self._load_external_plugins()
        
        # Set active profile
        profile_name = self.config.get_config("plugins.profile", "hill_climb_racing")
        self.profile_manager.set_active_profile(profile_name)
        
        log.info("Plugin system initialized")
    
    def _load_builtin_gestures(self):
        """Load built-in gesture plugins."""
        # Rule-based gestures
        rule_config = self.config.get_config("gestures", {})
        rule_plugin = RuleBasedGesturePlugin(rule_config)
        self.gesture_plugins["rule_based"] = rule_plugin
        
        # Add to active plugins
        active_plugins = self.config.get_config("plugins.gesture_plugins", "rule_based")
        self.active_gesture_plugins = [p.strip() for p in active_plugins.split(",") if p.strip()]
        
        log.info(f"Loaded built-in gesture plugins: {list(self.gesture_plugins.keys())}")
    
    def _load_external_plugins(self):
        """Load external gesture plugins from plugins directory."""
        plugins_dir = Path("plugins")
        if not plugins_dir.exists():
            return
        
        # Look for Python files in plugins directory
        for py_file in plugins_dir.glob("plugin_*.py"):
            try:
                module_name = f"plugins.{py_file.stem}"
                module = importlib.import_module(module_name)
                
                # Look for GesturePlugin classes
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and 
                        issubclass(attr, GesturePlugin) and 
                        attr != GesturePlugin):
                        
                        plugin_name = attr_name.lower().replace("gestureplugin", "")
                        plugin_config = self.config.get_config(f"plugins.{plugin_name}", {})
                        
                        self.gesture_plugins[plugin_name] = attr(plugin_config)
                        log.info(f"Loaded external plugin: {plugin_name}")
                        
            except Exception as e:
                log.error(f"Failed to load plugin {py_file}: {e}")
    
    def process_gestures(self, left_hand, right_hand, frame) -> List[GestureResult]:
        """Process hand landmarks through active gesture plugins."""
        all_results = []
        
        for plugin_name in self.active_gesture_plugins:
            if plugin_name not in self.gesture_plugins:
                continue
            
            plugin = self.gesture_plugins[plugin_name]
            if not plugin.enabled:
                continue
            
            try:
                # Process left hand
                if left_hand:
                    results = plugin.detect(left_hand, frame, "left")
                    all_results.extend(results)
                
                # Process right hand
                if right_hand:
                    results = plugin.detect(right_hand, frame, "right")
                    all_results.extend(results)
                    
            except Exception as e:
                log.error(f"Error in gesture plugin {plugin_name}: {e}")
        
        return all_results
    
    def execute_gestures(self, gesture_results: List[GestureResult]) -> Dict[str, Any]:
        """Execute actions based on detected gestures using active profile."""
        profile = self.profile_manager.get_active_profile()
        if not profile:
            return {}
        
        execution_results = {}
        
        for gesture_result in gesture_results:
            # Find matching mappings
            matching_mappings = self._find_matching_mappings(gesture_result, profile)
            
            for mapping in matching_mappings:
                try:
                    success = self.backend_manager.execute_action(
                        mapping.action_type,
                        mapping.action_params
                    )
                    
                    execution_results[gesture_result.gesture_name] = {
                        "success": success,
                        "action_type": mapping.action_type,
                        "action_params": mapping.action_params,
                        "confidence": gesture_result.confidence
                    }
                    
                except Exception as e:
                    log.error(f"Failed to execute action for {gesture_result.gesture_name}: {e}")
                    execution_results[gesture_result.gesture_name] = {
                        "success": False,
                        "error": str(e)
                    }
        
        return execution_results
    
    def _find_matching_mappings(self, gesture_result: GestureResult, profile: MappingProfile) -> List[ActionMapping]:
        """Find mappings that match the detected gesture."""
        matching = []
        
        for mapping in profile.mappings:
            if mapping.gesture == gesture_result.gesture_name:
                # Check conditions
                if self._check_mapping_conditions(gesture_result, mapping):
                    matching.append(mapping)
        
        return matching
    
    def _check_mapping_conditions(self, gesture_result: GestureResult, mapping: ActionMapping) -> bool:
        """Check if gesture result meets mapping conditions."""
        if not mapping.conditions:
            return True
        
        # Check confidence threshold
        min_confidence = mapping.conditions.get("confidence", 0.0)
        if gesture_result.confidence < min_confidence:
            return False
        
        # Check hand requirement
        required_hand = mapping.conditions.get("hand")
        if required_hand:
            gesture_hand = gesture_result.metadata.get("hand")
            if gesture_hand != required_hand:
                return False
        
        return True
    
    def _on_config_changed(self, new_config: Dict[str, Any]):
        """Handle configuration changes."""
        log.info("Configuration changed, updating plugins...")
        
        try:
            # Update gesture plugin configurations
            for name, plugin in self.gesture_plugins.items():
                if name == "rule_based":
                    plugin.configure(new_config.get("gestures", {}))
                else:
                    plugin.configure(new_config.get("plugins", {}).get(name, {}))
            
            # Update active plugins list
            active_plugins = new_config.get("plugins", {}).get("gesture_plugins", "rule_based")
            self.active_gesture_plugins = [p.strip() for p in active_plugins.split(",") if p.strip()]
            
            # Update active profile
            profile_name = new_config.get("plugins", {}).get("profile")
            if profile_name and profile_name != self.profile_manager.active_profile:
                self.profile_manager.set_active_profile(profile_name)
            
            # Update backend
            backend_name = new_config.get("backend", {}).get("actuation")
            if backend_name and backend_name != self.backend_manager.active_backend:
                self.backend_manager.set_active_backend(backend_name)
            
        except Exception as e:
            log.error(f"Error updating configuration: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current plugin system status."""
        return {
            "gesture_plugins": {
                "available": list(self.gesture_plugins.keys()),
                "active": self.active_gesture_plugins,
                "enabled": [name for name, plugin in self.gesture_plugins.items() if plugin.enabled]
            },
            "profiles": {
                "available": self.profile_manager.list_profiles(),
                "active": self.profile_manager.active_profile
            },
            "backends": {
                "available": self.backend_manager.list_backends(),
                "active": self.backend_manager.active_backend
            },
            "config_file": str(self.config.config_path),
            "config_watching": self.config._watching
        }
    
    def reload_profiles(self) -> List[str]:
        """Check for profile updates and reload if needed."""
        return self.profile_manager.check_for_updates()
    
    def create_profile(self, profile_data: Dict[str, Any]) -> bool:
        """Create a new mapping profile."""
        try:
            # Parse profile data into MappingProfile
            mappings = []
            for mapping_data in profile_data.get('mappings', []):
                mappings.append(ActionMapping(
                    gesture=mapping_data['gesture'],
                    action_type=mapping_data['action_type'],
                    action_params=mapping_data['action_params'],
                    conditions=mapping_data.get('conditions', {})
                ))
            
            profile = MappingProfile(
                name=profile_data['name'],
                description=profile_data.get('description', ''),
                target_application=profile_data.get('target_application', ''),
                mappings=mappings,
                settings=profile_data.get('settings', {}),
                version=profile_data.get('version', '1.0')
            )
            
            self.profile_manager.save_profile(profile)
            return True
            
        except Exception as e:
            log.error(f"Failed to create profile: {e}")
            return False
    
    def get_gesture_info(self) -> Dict[str, Any]:
        """Get information about available gestures."""
        gesture_info = {}
        
        for name, plugin in self.gesture_plugins.items():
            gesture_info[name] = {
                "gestures": plugin.get_gesture_names(),
                "enabled": plugin.enabled,
                "config_schema": plugin.get_config_schema()
            }
        
        return gesture_info
    
    def cleanup(self):
        """Clean up plugin system resources."""
        log.info("Cleaning up plugin system...")
        
        # Stop config watching
        self.config.stop_watching()
        
        # Cleanup backends
        self.backend_manager.cleanup_all()
        
        log.info("Plugin system cleanup complete")
