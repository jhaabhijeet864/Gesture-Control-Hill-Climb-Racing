"""
Actuation Backend System

Provides abstraction layer for different input methods:
- pynput (cross-platform keyboard/mouse)
- SendInput (Windows low-level)
- XInput/vgamepad (gamepad emulation)
- HID (hardware interface)
- OS Accessibility APIs
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import logging


log = logging.getLogger("actuation")


class ActuationBackend(ABC):
    """Base class for input actuation backends."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = self.__class__.__name__
        self.enabled = config.get('enabled', True)
        
    @abstractmethod
    def press_key(self, key: str, duration: Optional[float] = None) -> bool:
        """Press a key, optionally for a specific duration."""
        pass
    
    @abstractmethod
    def release_key(self, key: str) -> bool:
        """Release a key."""
        pass
    
    @abstractmethod
    def click_mouse(self, button: str, x: Optional[int] = None, y: Optional[int] = None) -> bool:
        """Click mouse button at optional coordinates."""
        pass
    
    @abstractmethod
    def move_mouse(self, x: int, y: int, relative: bool = False) -> bool:
        """Move mouse to coordinates."""
        pass
    
    def execute_action(self, action_type: str, params: Dict[str, Any]) -> bool:
        """Execute generic action based on type and parameters."""
        try:
            if action_type == "key":
                return self._handle_key_action(params)
            elif action_type == "mouse":
                return self._handle_mouse_action(params)
            elif action_type == "gamepad":
                return self._handle_gamepad_action(params)
            elif action_type == "sequence":
                return self._handle_sequence_action(params)
            else:
                log.warning(f"Unknown action type: {action_type}")
                return False
        except Exception as e:
            log.error(f"Action execution failed: {e}")
            return False
    
    def _handle_key_action(self, params: Dict[str, Any]) -> bool:
        """Handle keyboard action."""
        key = params.get('key')
        press_type = params.get('press_type', 'tap')  # tap, hold, release
        duration = params.get('duration')
        
        if press_type == 'tap':
            self.press_key(key, duration)
            if duration is None:
                self.release_key(key)
        elif press_type == 'hold':
            self.press_key(key)
        elif press_type == 'release':
            self.release_key(key)
        
        return True
    
    def _handle_mouse_action(self, params: Dict[str, Any]) -> bool:
        """Handle mouse action."""
        action = params.get('action')  # move, click, scroll
        
        if action == 'move':
            x = params.get('x', 0)
            y = params.get('y', 0)
            relative = params.get('relative', False)
            return self.move_mouse(x, y, relative)
        elif action == 'click':
            button = params.get('button', 'left')
            x = params.get('x')
            y = params.get('y')
            return self.click_mouse(button, x, y)
        
        return False
    
    def _handle_gamepad_action(self, params: Dict[str, Any]) -> bool:
        """Handle gamepad action (to be implemented by gamepad backends)."""
        log.warning("Gamepad actions not implemented in base backend")
        return False
    
    def _handle_sequence_action(self, params: Dict[str, Any]) -> bool:
        """Handle sequence of actions."""
        sequence = params.get('sequence', [])
        for action in sequence:
            if not self.execute_action(action.get('type'), action.get('params', {})):
                return False
        return True
    
    @abstractmethod
    def cleanup(self):
        """Clean up resources and release any held inputs."""
        pass


class PynputBackend(ActuationBackend):
    """Cross-platform backend using pynput."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        try:
            from pynput.keyboard import Controller as KeyboardController, Key
            from pynput.mouse import Controller as MouseController, Button
            
            self.keyboard = KeyboardController()
            self.mouse = MouseController()
            self.Key = Key
            self.Button = Button
            self.held_keys = set()
            self._key_mapping = self._create_key_mapping()
            
        except ImportError:
            log.error("pynput not available")
            self.enabled = False
    
    def _create_key_mapping(self) -> Dict[str, Any]:
        """Create mapping from string names to pynput keys."""
        mapping = {}
        
        # Add letter keys
        for char in 'abcdefghijklmnopqrstuvwxyz':
            mapping[char] = char
        
        # Add number keys
        for num in '0123456789':
            mapping[num] = num
        
        # Add special keys
        mapping.update({
            'space': self.Key.space,
            'enter': self.Key.enter,
            'tab': self.Key.tab,
            'escape': self.Key.esc,
            'shift': self.Key.shift,
            'ctrl': self.Key.ctrl,
            'alt': self.Key.alt,
            'up': self.Key.up,
            'down': self.Key.down,
            'left': self.Key.left,
            'right': self.Key.right,
            'f1': self.Key.f1, 'f2': self.Key.f2, 'f3': self.Key.f3, 'f4': self.Key.f4,
            'f5': self.Key.f5, 'f6': self.Key.f6, 'f7': self.Key.f7, 'f8': self.Key.f8,
            'f9': self.Key.f9, 'f10': self.Key.f10, 'f11': self.Key.f11, 'f12': self.Key.f12,
        })
        
        return mapping
    
    def press_key(self, key: str, duration: Optional[float] = None) -> bool:
        """Press a key."""
        if not self.enabled:
            return False
        
        try:
            pynput_key = self._key_mapping.get(key.lower(), key)
            self.keyboard.press(pynput_key)
            self.held_keys.add(key.lower())
            
            if duration:
                import time
                time.sleep(duration)
                self.release_key(key)
            
            return True
        except Exception as e:
            log.error(f"Failed to press key {key}: {e}")
            return False
    
    def release_key(self, key: str) -> bool:
        """Release a key."""
        if not self.enabled:
            return False
        
        try:
            pynput_key = self._key_mapping.get(key.lower(), key)
            self.keyboard.release(pynput_key)
            self.held_keys.discard(key.lower())
            return True
        except Exception as e:
            log.error(f"Failed to release key {key}: {e}")
            return False
    
    def click_mouse(self, button: str, x: Optional[int] = None, y: Optional[int] = None) -> bool:
        """Click mouse button."""
        if not self.enabled:
            return False
        
        try:
            if x is not None and y is not None:
                self.mouse.position = (x, y)
            
            mouse_button = getattr(self.Button, button.lower(), self.Button.left)
            self.mouse.click(mouse_button)
            return True
        except Exception as e:
            log.error(f"Failed to click mouse {button}: {e}")
            return False
    
    def move_mouse(self, x: int, y: int, relative: bool = False) -> bool:
        """Move mouse to coordinates."""
        if not self.enabled:
            return False
        
        try:
            if relative:
                current_x, current_y = self.mouse.position
                self.mouse.position = (current_x + x, current_y + y)
            else:
                self.mouse.position = (x, y)
            return True
        except Exception as e:
            log.error(f"Failed to move mouse: {e}")
            return False
    
    def cleanup(self):
        """Release all held keys."""
        for key in list(self.held_keys):
            self.release_key(key)


class SendInputBackend(ActuationBackend):
    """Windows SendInput backend for low-level input."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.held_keys = set()
        
        try:
            import platform
            if platform.system() != "Windows":
                log.warning("SendInput backend only works on Windows")
                self.enabled = False
                return
            
            # Import directkeys if available
            try:
                from directkeys import PressKey, ReleaseKey
                self.PressKey = PressKey
                self.ReleaseKey = ReleaseKey
                self._scan_codes = self._create_scan_code_mapping()
            except ImportError:
                log.warning("directkeys module not available")
                self.enabled = False
                
        except Exception as e:
            log.error(f"Failed to initialize SendInput backend: {e}")
            self.enabled = False
    
    def _create_scan_code_mapping(self) -> Dict[str, int]:
        """Create mapping from key names to scan codes."""
        return {
            'left': 0x4B,
            'right': 0x4D,
            'up': 0x48,
            'down': 0x50,
            'space': 0x39,
            'enter': 0x1C,
            'escape': 0x01,
            'a': 0x1E, 'b': 0x30, 'c': 0x2E, 'd': 0x20, 'e': 0x12,
            'f': 0x21, 'g': 0x22, 'h': 0x23, 'i': 0x17, 'j': 0x24,
            'k': 0x25, 'l': 0x26, 'm': 0x32, 'n': 0x31, 'o': 0x18,
            'p': 0x19, 'q': 0x10, 'r': 0x13, 's': 0x1F, 't': 0x14,
            'u': 0x16, 'v': 0x2F, 'w': 0x11, 'x': 0x2D, 'y': 0x15, 'z': 0x2C,
        }
    
    def press_key(self, key: str, duration: Optional[float] = None) -> bool:
        """Press a key using SendInput."""
        if not self.enabled:
            return False
        
        scan_code = self._scan_codes.get(key.lower())
        if scan_code is None:
            log.warning(f"Unknown key for SendInput: {key}")
            return False
        
        try:
            self.PressKey(scan_code)
            self.held_keys.add(key.lower())
            
            if duration:
                import time
                time.sleep(duration)
                self.release_key(key)
            
            return True
        except Exception as e:
            log.error(f"Failed to press key {key}: {e}")
            return False
    
    def release_key(self, key: str) -> bool:
        """Release a key using SendInput."""
        if not self.enabled:
            return False
        
        scan_code = self._scan_codes.get(key.lower())
        if scan_code is None:
            return False
        
        try:
            self.ReleaseKey(scan_code)
            self.held_keys.discard(key.lower())
            return True
        except Exception as e:
            log.error(f"Failed to release key {key}: {e}")
            return False
    
    def click_mouse(self, button: str, x: Optional[int] = None, y: Optional[int] = None) -> bool:
        """Mouse clicks not implemented in SendInput backend."""
        log.warning("Mouse actions not implemented in SendInput backend")
        return False
    
    def move_mouse(self, x: int, y: int, relative: bool = False) -> bool:
        """Mouse movement not implemented in SendInput backend."""
        log.warning("Mouse actions not implemented in SendInput backend")
        return False
    
    def cleanup(self):
        """Release all held keys."""
        for key in list(self.held_keys):
            self.release_key(key)


class BackendManager:
    """Manages multiple actuation backends."""
    
    def __init__(self):
        self.backends: Dict[str, ActuationBackend] = {}
        self.active_backend: Optional[str] = None
        
    def register_backend(self, name: str, backend: ActuationBackend):
        """Register a new backend."""
        self.backends[name] = backend
        if self.active_backend is None and backend.enabled:
            self.active_backend = name
    
    def set_active_backend(self, name: str) -> bool:
        """Set the active backend."""
        if name in self.backends and self.backends[name].enabled:
            self.active_backend = name
            return True
        return False
    
    def get_active_backend(self) -> Optional[ActuationBackend]:
        """Get the currently active backend."""
        if self.active_backend:
            return self.backends.get(self.active_backend)
        return None
    
    def execute_action(self, action_type: str, params: Dict[str, Any]) -> bool:
        """Execute action using active backend."""
        backend = self.get_active_backend()
        if backend:
            return backend.execute_action(action_type, params)
        return False
    
    def list_backends(self) -> List[str]:
        """List all available backends."""
        return [name for name, backend in self.backends.items() if backend.enabled]
    
    def cleanup_all(self):
        """Cleanup all backends."""
        for backend in self.backends.values():
            try:
                backend.cleanup()
            except Exception as e:
                log.error(f"Error cleaning up backend: {e}")
                
    def create_default_backends(self, config: Dict[str, Any]) -> List[str]:
        """Create and register default backends."""
        created = []
        
        # Pynput backend (cross-platform)
        pynput_config = config.get('pynput', {'enabled': True})
        pynput_backend = PynputBackend(pynput_config)
        if pynput_backend.enabled:
            self.register_backend('pynput', pynput_backend)
            created.append('pynput')
        
        # SendInput backend (Windows)
        sendinput_config = config.get('sendinput', {'enabled': True})
        sendinput_backend = SendInputBackend(sendinput_config)
        if sendinput_backend.enabled:
            self.register_backend('sendinput', sendinput_backend)
            created.append('sendinput')
        
        return created
