"""
Configuration Management with Live Reload

Enhanced configuration system supporting YAML/JSON with hot-reload, 
UI slider generation, and validation.
"""
import json
import yaml
import os
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
import time
import logging
from dataclasses import dataclass, asdict
from threading import Thread, Event
import copy


log = logging.getLogger("config")


@dataclass
class ConfigSchema:
    """Schema definition for configuration parameters."""
    param_type: str  # "float", "int", "boolean", "string", "enum"
    default: Any
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    options: Optional[List[Any]] = None  # For enum type
    description: str = ""
    category: str = "general"
    requires_restart: bool = False


class LiveConfigManager:
    """Configuration manager with hot-reload and validation."""
    
    def __init__(self, config_path: str = "config.yaml", watch_interval: float = 1.0):
        self.config_path = Path(config_path)
        self.watch_interval = watch_interval
        self.config_data: Dict[str, Any] = {}
        self.schemas: Dict[str, ConfigSchema] = {}
        self.last_modified = 0
        self.watchers: List[Callable[[Dict[str, Any]], None]] = []
        
        # Threading for file watching
        self._stop_watching = Event()
        self._watch_thread: Optional[Thread] = None
        self._watching = False
        
        # Load initial config
        self.load_config()
        
    def register_schema(self, path: str, schema: ConfigSchema):
        """Register schema for a configuration parameter."""
        self.schemas[path] = schema
        
        # Set default value if not present
        if not self.has_config(path):
            self.set_config(path, schema.default, save=False)
    
    def register_schemas(self, schemas: Dict[str, ConfigSchema]):
        """Register multiple schemas at once."""
        for path, schema in schemas.items():
            self.register_schema(path, schema)
    
    def get_config(self, path: str, default: Any = None) -> Any:
        """Get configuration value by dot-separated path."""
        try:
            keys = path.split('.')
            value = self.config_data
            
            for key in keys:
                value = value[key]
            
            return value
        except (KeyError, TypeError):
            return default
    
    def set_config(self, path: str, value: Any, save: bool = True):
        """Set configuration value by dot-separated path."""
        keys = path.split('.')
        config = self.config_data
        
        # Navigate to parent dict
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # Validate value against schema
        if path in self.schemas:
            value = self._validate_value(value, self.schemas[path])
        
        # Set the value
        config[keys[-1]] = value
        
        if save:
            self.save_config()
        
        # Notify watchers
        self._notify_watchers()
    
    def has_config(self, path: str) -> bool:
        """Check if configuration path exists."""
        try:
            keys = path.split('.')
            value = self.config_data
            
            for key in keys:
                value = value[key]
            
            return True
        except (KeyError, TypeError):
            return False
    
    def _validate_value(self, value: Any, schema: ConfigSchema) -> Any:
        """Validate and convert value according to schema."""
        try:
            if schema.param_type == "float":
                value = float(value)
                if schema.min_value is not None:
                    value = max(schema.min_value, value)
                if schema.max_value is not None:
                    value = min(schema.max_value, value)
            
            elif schema.param_type == "int":
                value = int(value)
                if schema.min_value is not None:
                    value = max(int(schema.min_value), value)
                if schema.max_value is not None:
                    value = min(int(schema.max_value), value)
            
            elif schema.param_type == "boolean":
                if isinstance(value, str):
                    value = value.lower() in ("true", "1", "yes", "on")
                else:
                    value = bool(value)
            
            elif schema.param_type == "enum":
                if schema.options and value not in schema.options:
                    log.warning(f"Invalid enum value {value}, using default {schema.default}")
                    value = schema.default
            
            return value
            
        except (ValueError, TypeError) as e:
            log.error(f"Validation failed for {value}: {e}")
            return schema.default
    
    def load_config(self):
        """Load configuration from file."""
        if not self.config_path.exists():
            # Create default config
            self.config_data = self._create_default_config()
            self.save_config()
        else:
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    if self.config_path.suffix.lower() == '.yaml':
                        self.config_data = yaml.safe_load(f) or {}
                    else:
                        self.config_data = json.load(f)
                
                self.last_modified = self.config_path.stat().st_mtime
                log.info(f"Loaded config from {self.config_path}")
                
            except Exception as e:
                log.error(f"Failed to load config: {e}")
                self.config_data = self._create_default_config()
    
    def save_config(self):
        """Save configuration to file."""
        try:
            # Create backup
            if self.config_path.exists():
                backup_path = self.config_path.with_suffix(f"{self.config_path.suffix}.bak")
                backup_path.write_bytes(self.config_path.read_bytes())
            
            # Save new config
            with open(self.config_path, 'w', encoding='utf-8') as f:
                if self.config_path.suffix.lower() == '.yaml':
                    yaml.dump(self.config_data, f, default_flow_style=False, indent=2)
                else:
                    json.dump(self.config_data, f, indent=2)
            
            self.last_modified = self.config_path.stat().st_mtime
            log.info(f"Saved config to {self.config_path}")
            
        except Exception as e:
            log.error(f"Failed to save config: {e}")
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration from schemas."""
        config = {}
        
        for path, schema in self.schemas.items():
            keys = path.split('.')
            current = config
            
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            
            current[keys[-1]] = schema.default
        
        return config
    
    def add_watcher(self, callback: Callable[[Dict[str, Any]], None]):
        """Add a callback that gets called when config changes."""
        self.watchers.append(callback)
    
    def remove_watcher(self, callback: Callable[[Dict[str, Any]], None]):
        """Remove a config change callback."""
        if callback in self.watchers:
            self.watchers.remove(callback)
    
    def _notify_watchers(self):
        """Notify all watchers of config changes."""
        for watcher in self.watchers:
            try:
                watcher(copy.deepcopy(self.config_data))
            except Exception as e:
                log.error(f"Config watcher error: {e}")
    
    def start_watching(self):
        """Start watching config file for changes."""
        if self._watching:
            return
        
        self._watching = True
        self._stop_watching.clear()
        self._watch_thread = Thread(target=self._watch_loop, daemon=True)
        self._watch_thread.start()
        log.info("Started config file watching")
    
    def stop_watching(self):
        """Stop watching config file."""
        if not self._watching:
            return
        
        self._watching = False
        self._stop_watching.set()
        
        if self._watch_thread:
            self._watch_thread.join(timeout=2.0)
        
        log.info("Stopped config file watching")
    
    def _watch_loop(self):
        """Main loop for watching config file changes."""
        while not self._stop_watching.wait(self.watch_interval):
            try:
                if self.config_path.exists():
                    current_mtime = self.config_path.stat().st_mtime
                    
                    if current_mtime > self.last_modified:
                        log.info("Config file changed, reloading...")
                        old_config = copy.deepcopy(self.config_data)
                        self.load_config()
                        
                        # Check if anything actually changed
                        if self.config_data != old_config:
                            self._notify_watchers()
                        
            except Exception as e:
                log.error(f"Error in config watch loop: {e}")
    
    def get_ui_config(self) -> Dict[str, Any]:
        """Get configuration formatted for UI generation."""
        ui_config = {}
        
        for path, schema in self.schemas.items():
            current_value = self.get_config(path, schema.default)
            
            ui_config[path] = {
                "value": current_value,
                "type": schema.param_type,
                "min": schema.min_value,
                "max": schema.max_value,
                "step": schema.step,
                "options": schema.options,
                "description": schema.description,
                "category": schema.category,
                "requires_restart": schema.requires_restart
            }
        
        return ui_config
    
    def export_config(self, path: str, format: str = "yaml"):
        """Export current configuration to file."""
        export_path = Path(path)
        
        try:
            with open(export_path, 'w', encoding='utf-8') as f:
                if format.lower() == 'yaml':
                    yaml.dump(self.config_data, f, default_flow_style=False, indent=2)
                else:
                    json.dump(self.config_data, f, indent=2)
            
            log.info(f"Exported config to {export_path}")
        except Exception as e:
            log.error(f"Failed to export config: {e}")
    
    def import_config(self, path: str):
        """Import configuration from file."""
        import_path = Path(path)
        
        if not import_path.exists():
            log.error(f"Import file does not exist: {import_path}")
            return
        
        try:
            with open(import_path, 'r', encoding='utf-8') as f:
                if import_path.suffix.lower() == '.yaml':
                    imported_data = yaml.safe_load(f)
                else:
                    imported_data = json.load(f)
            
            # Merge with current config
            self._deep_merge(self.config_data, imported_data)
            self.save_config()
            self._notify_watchers()
            
            log.info(f"Imported config from {import_path}")
            
        except Exception as e:
            log.error(f"Failed to import config: {e}")
    
    def _deep_merge(self, target: Dict[str, Any], source: Dict[str, Any]):
        """Deep merge source dict into target dict."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value


def create_default_schemas() -> Dict[str, ConfigSchema]:
    """Create default configuration schemas for Game Glide."""
    return {
        # Core detection settings
        "detection.min_detection_confidence": ConfigSchema(
            "float", 0.7, 0.1, 1.0, 0.05,
            description="Minimum confidence for hand detection",
            category="Detection"
        ),
        "detection.min_tracking_confidence": ConfigSchema(
            "float", 0.7, 0.1, 1.0, 0.05,
            description="Minimum confidence for hand tracking",
            category="Detection"
        ),
        "detection.max_num_hands": ConfigSchema(
            "int", 2, 1, 4, 1,
            description="Maximum number of hands to detect",
            category="Detection"
        ),
        
        # Gesture thresholds
        "gestures.fist_threshold": ConfigSchema(
            "float", 0.8, 0.1, 1.0, 0.05,
            description="Threshold for fist detection",
            category="Gestures"
        ),
        "gestures.pinch_threshold": ConfigSchema(
            "float", 0.05, 0.01, 0.2, 0.005,
            description="Distance threshold for pinch detection",
            category="Gestures"
        ),
        "gestures.point_threshold": ConfigSchema(
            "float", 0.7, 0.1, 1.0, 0.05,
            description="Confidence threshold for pointing",
            category="Gestures"
        ),
        
        # UI and performance
        "ui.smoothing": ConfigSchema(
            "int", 8, 1, 32, 1,
            description="Cursor movement smoothing factor",
            category="UI"
        ),
        "ui.show_fps": ConfigSchema(
            "boolean", True,
            description="Show FPS counter",
            category="UI"
        ),
        "ui.show_landmarks": ConfigSchema(
            "boolean", True,
            description="Show hand landmarks",
            category="UI"
        ),
        
        # Backend selection
        "backend.actuation": ConfigSchema(
            "enum", "pynput", options=["pynput", "sendinput"],
            description="Input backend to use",
            category="Backend",
            requires_restart=True
        ),
        
        # Plugin settings
        "plugins.gesture_plugins": ConfigSchema(
            "string", "rule_based",
            description="Active gesture plugins (comma-separated)",
            category="Plugins"
        ),
        "plugins.profile": ConfigSchema(
            "string", "hill_climb_racing",
            description="Active mapping profile",
            category="Plugins"
        )
    }
