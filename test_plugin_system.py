#!/usr/bin/env python3
"""
Test script for the Game Glide plugin system
"""

import sys
import os
import logging

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from plugins.plugin_manager import PluginManager
from plugins.live_config import LiveConfigManager
from plugins.mapping_profiles import ProfileManager
from plugins.actuation_backends import BackendManager

def test_config_loading():
    """Test configuration loading"""
    print("Testing configuration loading...")
    try:
        config_manager = LiveConfigManager()
        config = config_manager.get_config("detection.min_detection_confidence", 0.7)
        print(f"✓ Config loaded successfully, detection confidence: {config}")
        return True
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False

def test_profile_loading():
    """Test profile loading"""
    print("\nTesting profile loading...")
    try:
        profile_manager = ProfileManager()
        profiles = profile_manager.list_profiles()
        print(f"✓ Found profiles: {profiles}")
        
        if profiles:
            profile = profile_manager.load_profile(profiles[0])
            print(f"✓ Loaded profile '{profiles[0]}' with {len(profile.mappings)} mappings")
        return True
    except Exception as e:
        print(f"✗ Profile loading failed: {e}")
        return False

def test_backend_initialization():
    """Test backend initialization"""
    print("\nTesting backend initialization...")
    try:
        backend_manager = BackendManager()
        backend_manager.set_active_backend("pynput")
        print("✓ Backend manager initialized with pynput")
        return True
    except Exception as e:
        print(f"✗ Backend initialization failed: {e}")
        return False

def test_plugin_manager():
    """Test plugin manager"""
    print("\nTesting plugin manager...")
    try:
        plugin_manager = PluginManager()
        print("✓ Plugin manager created successfully")
        
        # Test gesture processing with dummy landmarks
        dummy_landmarks = [[0.5, 0.5, 0.0] for _ in range(21)]  # Dummy hand landmarks
        import numpy as np
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)  # Dummy frame
        results = plugin_manager.process_gestures(dummy_landmarks, None, dummy_frame)
        print(f"✓ Gesture processing works, results: {len(results)} gestures detected")
        return True
    except Exception as e:
        print(f"✗ Plugin manager failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Game Glide Plugin System Test")
    print("=" * 40)
    
    # Suppress MediaPipe logs for cleaner output
    logging.getLogger('mediapipe').setLevel(logging.WARNING)
    
    tests = [
        test_config_loading,
        test_profile_loading,
        test_backend_initialization,
        test_plugin_manager
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
    
    print("\n" + "=" * 40)
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Plugin system is ready.")
        return True
    else:
        print("❌ Some tests failed. Check the output above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
