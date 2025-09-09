#!/usr/bin/env python3
"""
Test script to verify ML gesture recognition system components
"""

import sys
import os
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_feature_extraction():
    """Test feature extraction."""
    print("🧪 Testing Feature Extraction...")
    
    try:
        from ml_gesture_recognition import FeatureExtractor
        
        # Create dummy landmarks (21 landmarks for MediaPipe hands)
        dummy_landmarks = [(0.5 + i*0.01, 0.5 + i*0.01, 0.0) for i in range(21)]
        
        extractor = FeatureExtractor()
        features = extractor.extract_features(dummy_landmarks)
        
        print(f"✅ Feature extraction successful!")
        print(f"   - Type: {type(features)}")
        print(f"   - Has finger_angles: {hasattr(features, 'finger_angles')}")
        print(f"   - Has finger_distances: {hasattr(features, 'finger_distances')}")
        print(f"   - Detection confidence: {features.detection_confidence}")
        
        return True
    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")
        return False

def test_calibration_system():
    """Test calibration system."""
    print("\n🧪 Testing Calibration System...")
    
    try:
        from calibration_filtering import GestureCalibrationFlow
        from ml_gesture_recognition import FeatureExtractor
        
        extractor = FeatureExtractor()
        calibration = GestureCalibrationFlow(extractor)
        
        # Test calibration start
        result = calibration.start_calibration("test_user")
        print(f"✅ Calibration start successful!")
        print(f"   - Status: {result['status']}")
        print(f"   - Current stage: {result['current_stage']['name']}")
        
        return True
    except Exception as e:
        print(f"❌ Calibration system failed: {e}")
        return False

def test_ml_classifier():
    """Test ML classifier."""
    print("\n🧪 Testing ML Classifier...")
    
    try:
        from ml_gesture_recognition import MLGestureClassifier
        
        classifier = MLGestureClassifier()
        
        # Create dummy training data
        features = np.random.random((100, 50)).tolist()  # 100 samples, 50 features
        labels = ["gesture_1"] * 50 + ["gesture_2"] * 50
        
        print("✅ ML classifier initialized!")
        print(f"   - Training data: {len(features)} samples")
        print(f"   - Labels: {set(labels)}")
        
        return True
    except Exception as e:
        print(f"❌ ML classifier failed: {e}")
        return False

def test_temporal_models():
    """Test temporal models."""
    print("\n🧪 Testing Temporal Models...")
    
    try:
        from temporal_gesture_models import TemporalGestureRecognizer
        
        recognizer = TemporalGestureRecognizer(model_type="lstm")
        
        print("✅ Temporal model initialized!")
        print(f"   - Model type: lstm")
        print(f"   - Device: {recognizer.device}")
        print(f"   - Max sequence length: {recognizer.max_sequence_length}")
        
        return True
    except Exception as e:
        print(f"❌ Temporal models failed: {e}")
        return False

def test_robust_system():
    """Test the integrated robust system."""
    print("\n🧪 Testing Robust Gesture System...")
    
    try:
        from robust_gesture_system import RobustGestureRecognitionSystem
        
        system = RobustGestureRecognitionSystem()
        
        # Test system status
        status = system.get_system_status()
        
        print("✅ Robust system initialized!")
        print(f"   - Calibrated: {status['calibrated']}")
        print(f"   - ML trained: {status['ml_trained']}")
        print(f"   - Temporal trained: {status['temporal_trained']}")
        print(f"   - Config: {len(status['config'])} parameters")
        
        # Test feature extraction with dummy data
        dummy_landmarks = [(0.5 + i*0.01, 0.5 + i*0.01, 0.0) for i in range(21)]
        
        # Extract features using the robust system
        gesture_features = system.feature_extractor.extract_features(dummy_landmarks)
        numerical_features = system._extract_numerical_features(gesture_features)
        
        print(f"   - Numerical features extracted: {len(numerical_features)} features")
        
        return True
    except Exception as e:
        print(f"❌ Robust system failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_filtering_system():
    """Test adaptive filtering system."""
    print("\n🧪 Testing Filtering System...")
    
    try:
        from calibration_filtering import OneEuroFilter, KalmanFilter, AdaptiveGestureFilter
        
        # Test One Euro Filter
        one_euro = OneEuroFilter()
        filtered_value = one_euro(1.0)
        print(f"✅ One Euro Filter: {filtered_value}")
        
        # Test Kalman Filter
        kalman = KalmanFilter()
        filtered_value = kalman.update(1.0)
        print(f"✅ Kalman Filter: {filtered_value}")
        
        # Test Adaptive Filter
        adaptive = AdaptiveGestureFilter()
        adaptive.initialize_filters(["feature_1", "feature_2"])
        
        print("✅ Filtering system working!")
        return True
    except Exception as e:
        print(f"❌ Filtering system failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 ML Gesture Recognition System Test Suite")
    print("=" * 50)
    
    tests = [
        test_feature_extraction,
        test_calibration_system,
        test_ml_classifier,
        test_temporal_models,
        test_robust_system,
        test_filtering_system
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        print("\n🎯 Next steps:")
        print("   1. Run 'python ml_gesture_demo.py' to start the demo")
        print("   2. Press 'C' to calibrate the system")
        print("   3. Press 'T' to train custom gestures")
        print("   4. Press 'R' for real-time recognition")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
