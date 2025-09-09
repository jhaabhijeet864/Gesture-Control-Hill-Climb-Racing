#!/usr/bin/env python3
"""
Game Glide v1.0.0 - Quick Installation Script
Automated setup for Game Glide gesture recognition system
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle output."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True
        else:
            print(f"❌ {description} failed:")
            print(f"   {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {description} failed: {e}")
        return False

def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing Dependencies")
    print("-" * 30)
    
    # Basic packages
    packages = [
        "opencv-python>=4.8.0",
        "mediapipe>=0.10.0", 
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "scipy>=1.11.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "tqdm>=4.65.0",
        "joblib>=1.3.0"
    ]
    
    for package in packages:
        if not run_command(f"pip install {package}", f"Installing {package.split('>=')[0]}"):
            return False
    
    return True

def verify_installation():
    """Verify installation by running system check."""
    print("\n🧪 Verifying Installation")
    print("-" * 30)
    
    return run_command("python setup_check.py", "Running system verification")

def main():
    """Main installation process."""
    print("🎯 Game Glide v1.0.0 - Installation Script")
    print("=" * 50)
    print("This script will install all required dependencies for Game Glide.")
    print("Make sure you have Python 3.8+ installed.\n")
    
    # Check Python version
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ required")
        print(f"   Current version: {version.major}.{version.minor}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor} detected")
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Installation failed. Please check error messages above.")
        return False
    
    # Verify installation
    if not verify_installation():
        print("\n⚠️  Installation completed but verification failed.")
        print("   You may need to resolve some issues manually.")
        return False
    
    print("\n🎉 Installation completed successfully!")
    print("\n🚀 Next Steps:")
    print("   1. Run: python ml_gesture_demo.py")
    print("   2. Press 'C' to calibrate")
    print("   3. Press 'T' to train models") 
    print("   4. Press 'R' for recognition")
    print("\n📚 Documentation: ML_GESTURE_README.md")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
