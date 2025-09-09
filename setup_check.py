#!/usr/bin/env python3
"""
Game Glide v1.0.0 - Quick Setup Verification
Verifies installation and system readiness
"""

import sys
import os
import importlib
from typing import List, Tuple

def check_python_version() -> Tuple[bool, str]:
    """Check Python version compatibility."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        return True, f"✅ Python {version.major}.{version.minor}.{version.micro}"
    else:
        return False, f"❌ Python {version.major}.{version.minor} (requires 3.8+)"

def check_dependencies() -> List[Tuple[str, bool, str]]:
    """Check required dependencies."""
    required_packages = [
        "cv2",
        "mediapipe", 
        "numpy",
        "sklearn",
        "torch",
        "pandas",
        "matplotlib",
        "scipy",
        "tkinter"
    ]
    
    results = []
    for package in required_packages:
        try:
            if package == "cv2":
                import cv2
                version = cv2.__version__
            elif package == "tkinter":
                import tkinter
                version = "built-in"
            else:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'unknown')
            
            results.append((package, True, f"✅ {package} {version}"))
        except ImportError:
            results.append((package, False, f"❌ {package} - not installed"))
    
    return results

def check_system_components() -> List[Tuple[str, bool, str]]:
    """Check Game Glide system components."""
    components = [
        "robust_gesture_system.py",
        "ml_gesture_recognition.py", 
        "temporal_gesture_models.py",
        "calibration_filtering.py",
        "ml_gesture_demo.py",
        "requirements.txt",
        "ML_GESTURE_README.md"
    ]
    
    results = []
    for component in components:
        if os.path.exists(component):
            results.append((component, True, f"✅ {component}"))
        else:
            results.append((component, False, f"❌ {component} - missing"))
    
    return results

def check_camera_access() -> Tuple[bool, str]:
    """Check camera accessibility."""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                return True, "✅ Camera accessible"
            else:
                return False, "❌ Camera detected but no frame captured"
        else:
            return False, "❌ No camera detected"
    except Exception as e:
        return False, f"❌ Camera check failed: {str(e)}"

def main():
    """Run complete system verification."""
    print("🎯 Game Glide v1.0.0 - System Verification")
    print("=" * 50)
    
    # Check Python version
    python_ok, python_msg = check_python_version()
    print(f"\n🐍 Python Version:")
    print(f"   {python_msg}")
    
    # Check dependencies
    print(f"\n📦 Dependencies:")
    deps = check_dependencies()
    deps_ok = all(ok for _, ok, _ in deps)
    
    for _, ok, msg in deps:
        print(f"   {msg}")
    
    # Check components
    print(f"\n🔧 System Components:")
    components = check_system_components()
    components_ok = all(ok for _, ok, _ in components)
    
    for _, ok, msg in components:
        print(f"   {msg}")
    
    # Check camera
    print(f"\n📷 Camera Access:")
    camera_ok, camera_msg = check_camera_access()
    print(f"   {camera_msg}")
    
    # Overall status
    print(f"\n📊 System Status:")
    all_good = python_ok and deps_ok and components_ok and camera_ok
    
    if all_good:
        print("   🎉 All systems ready! Game Glide is ready to run.")
        print("\n🚀 Quick Start:")
        print("   python ml_gesture_demo.py")
        print("\n📚 Documentation:")
        print("   See ML_GESTURE_README.md for detailed usage")
    else:
        print("   ⚠️  Some issues detected. Please resolve them before running.")
        
        if not python_ok:
            print("   📌 Update Python to version 3.8 or higher")
        
        if not deps_ok:
            print("   📌 Install missing dependencies: pip install -r requirements.txt")
        
        if not components_ok:
            print("   📌 Download missing components from the repository")
        
        if not camera_ok:
            print("   📌 Check camera permissions and connectivity")
    
    return all_good

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
