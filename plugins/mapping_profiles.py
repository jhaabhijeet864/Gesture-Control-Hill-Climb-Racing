"""
Mapping Profile System

Handles per-game configuration profiles that map gestures to actions.
Supports JSON/YAML configs with hot-reload capability.
"""
import json
import yaml
import os
from typing import Dict, Any, List, Optional
from pathlib import Path
import time
from dataclasses import dataclass


@dataclass
class ActionMapping:
    """Single gesture-to-action mapping."""
    gesture: str
    action_type: str  # 'key', 'mouse', 'gamepad', 'sequence'
    action_params: Dict[str, Any]
    conditions: Dict[str, Any] = None  # Optional conditions (hand, confidence, etc.)


@dataclass
class MappingProfile:
    """Complete mapping profile for a game/application."""
    name: str
    description: str
    target_application: str
    mappings: List[ActionMapping]
    settings: Dict[str, Any]
    version: str = "1.0"


class ProfileManager:
    """Manages loading, saving, and hot-reloading of mapping profiles."""
    
    def __init__(self, profiles_dir: str = "profiles"):
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(exist_ok=True)
        self.profiles: Dict[str, MappingProfile] = {}
        self.file_timestamps: Dict[str, float] = {}
        self.active_profile: Optional[str] = None
        
    def load_profile(self, profile_name: str) -> Optional[MappingProfile]:
        """Load a profile from file."""
        profile_path = self.profiles_dir / f"{profile_name}.json"
        yaml_path = self.profiles_dir / f"{profile_name}.yaml"
        
        # Try JSON first, then YAML
        if profile_path.exists():
            return self._load_json_profile(profile_path, profile_name)
        elif yaml_path.exists():
            return self._load_yaml_profile(yaml_path, profile_name)
        else:
            return None
    
    def _load_json_profile(self, path: Path, profile_name: str) -> MappingProfile:
        """Load profile from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.file_timestamps[profile_name] = path.stat().st_mtime
        return self._parse_profile_data(data, profile_name)
    
    def _load_yaml_profile(self, path: Path, profile_name: str) -> MappingProfile:
        """Load profile from YAML file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        self.file_timestamps[profile_name] = path.stat().st_mtime
        return self._parse_profile_data(data, profile_name)
    
    def _parse_profile_data(self, data: Dict[str, Any], profile_name: str) -> MappingProfile:
        """Parse profile data into MappingProfile object."""
        mappings = []
        for mapping_data in data.get('mappings', []):
            mappings.append(ActionMapping(
                gesture=mapping_data['gesture'],
                action_type=mapping_data['action_type'],
                action_params=mapping_data['action_params'],
                conditions=mapping_data.get('conditions', {})
            ))
        
        return MappingProfile(
            name=data.get('name', profile_name),
            description=data.get('description', ''),
            target_application=data.get('target_application', ''),
            mappings=mappings,
            settings=data.get('settings', {}),
            version=data.get('version', '1.0')
        )
    
    def save_profile(self, profile: MappingProfile, format: str = 'json'):
        """Save profile to file."""
        if format == 'json':
            path = self.profiles_dir / f"{profile.name}.json"
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self._profile_to_dict(profile), f, indent=2)
        elif format == 'yaml':
            path = self.profiles_dir / f"{profile.name}.yaml"
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(self._profile_to_dict(profile), f, default_flow_style=False)
    
    def _profile_to_dict(self, profile: MappingProfile) -> Dict[str, Any]:
        """Convert MappingProfile to dictionary for serialization."""
        return {
            'name': profile.name,
            'description': profile.description,
            'target_application': profile.target_application,
            'version': profile.version,
            'settings': profile.settings,
            'mappings': [
                {
                    'gesture': m.gesture,
                    'action_type': m.action_type,
                    'action_params': m.action_params,
                    'conditions': m.conditions or {}
                }
                for m in profile.mappings
            ]
        }
    
    def list_profiles(self) -> List[str]:
        """List all available profiles."""
        profiles = []
        for path in self.profiles_dir.glob("*.json"):
            profiles.append(path.stem)
        for path in self.profiles_dir.glob("*.yaml"):
            if path.stem not in profiles:  # Avoid duplicates
                profiles.append(path.stem)
        return sorted(profiles)
    
    def set_active_profile(self, profile_name: str) -> bool:
        """Set the active profile."""
        if profile_name in self.list_profiles():
            self.active_profile = profile_name
            if profile_name not in self.profiles:
                self.profiles[profile_name] = self.load_profile(profile_name)
            return True
        return False
    
    def get_active_profile(self) -> Optional[MappingProfile]:
        """Get the currently active profile."""
        if self.active_profile:
            return self.profiles.get(self.active_profile)
        return None
    
    def check_for_updates(self) -> List[str]:
        """Check for profile file updates and reload if needed."""
        updated_profiles = []
        
        for profile_name in list(self.profiles.keys()):
            json_path = self.profiles_dir / f"{profile_name}.json"
            yaml_path = self.profiles_dir / f"{profile_name}.yaml"
            
            current_path = json_path if json_path.exists() else yaml_path
            
            if current_path.exists():
                current_mtime = current_path.stat().st_mtime
                stored_mtime = self.file_timestamps.get(profile_name, 0)
                
                if current_mtime > stored_mtime:
                    # Reload profile
                    updated_profile = self.load_profile(profile_name)
                    if updated_profile:
                        self.profiles[profile_name] = updated_profile
                        updated_profiles.append(profile_name)
        
        return updated_profiles
    
    def create_default_profiles(self):
        """Create default example profiles."""
        # Hill Climb Racing profile
        hill_climb_profile = MappingProfile(
            name="hill_climb_racing",
            description="Profile for Hill Climb Racing game",
            target_application="Hill Climb Racing",
            settings={
                "smoothing": 8,
                "confidence_threshold": 0.7,
                "gesture_cooldown": 0.1
            },
            mappings=[
                ActionMapping(
                    gesture="fist_right",
                    action_type="key",
                    action_params={"key": "right", "press_type": "hold"},
                    conditions={"confidence": 0.8}
                ),
                ActionMapping(
                    gesture="fist_left",
                    action_type="key",
                    action_params={"key": "left", "press_type": "hold"},
                    conditions={"confidence": 0.8}
                )
            ]
        )
        
        # Mouse control profile
        mouse_profile = MappingProfile(
            name="mouse_control",
            description="General mouse and cursor control",
            target_application="Desktop",
            settings={
                "cursor_smoothing": 5,
                "click_threshold": 0.05,
                "scroll_sensitivity": 3
            },
            mappings=[
                ActionMapping(
                    gesture="point",
                    action_type="mouse",
                    action_params={"action": "move", "hand": "left"}
                ),
                ActionMapping(
                    gesture="pinch_thumb_index",
                    action_type="mouse",
                    action_params={"action": "click", "button": "left"},
                    conditions={"hand": "left", "confidence": 0.7}
                ),
                ActionMapping(
                    gesture="pinch_thumb_middle",
                    action_type="mouse",
                    action_params={"action": "click", "button": "right"},
                    conditions={"hand": "left", "confidence": 0.7}
                )
            ]
        )
        
        # Save default profiles
        self.save_profile(hill_climb_profile, 'json')
        self.save_profile(mouse_profile, 'json')
        
        return [hill_climb_profile.name, mouse_profile.name]
