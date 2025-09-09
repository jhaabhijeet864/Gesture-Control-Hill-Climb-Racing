#!/usr/bin/env python3
"""
Gesture Calibration and Filtering System
"""

import numpy as np
import cv2
import mediapipe as mp
import json
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import deque
import time
import statistics
from scipy.spatial.distance import euclidean
from scipy.signal import butter, filtfilt
import logging

@dataclass
class HandCalibrationData:
    """Calibration data for a user's hand."""
    hand_size: float                    # Average hand span
    finger_lengths: Dict[str, float]    # Individual finger lengths
    palm_size: float                    # Palm area/size
    joint_flexibility: Dict[str, float] # Range of motion for joints
    gesture_thresholds: Dict[str, float] # Gesture-specific thresholds
    lighting_adaptation: Dict[str, float] # Lighting condition adaptations
    user_id: str = "default"
    calibration_date: str = ""
    num_calibration_samples: int = 0

class OneEuroFilter:
    """One Euro Filter for smooth gesture tracking."""
    
    def __init__(self, freq: float = 30.0, mincutoff: float = 1.0, 
                 beta: float = 0.007, dcutoff: float = 1.0):
        """
        Initialize One Euro Filter.
        
        Args:
            freq: Sampling frequency
            mincutoff: Minimum cutoff frequency
            beta: Cutoff slope
            dcutoff: Cutoff frequency for derivative
        """
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        
        self.x_prev = None
        self.dx_prev = None
        self.t_prev = None

    def __call__(self, x: float, t: Optional[float] = None) -> float:
        """Apply filter to new measurement."""
        if t is None:
            t = time.time()
        
        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            self.t_prev = t
            return x
        
        # Calculate time delta
        dt = t - self.t_prev
        if dt <= 0:
            return self.x_prev
        
        # Update frequency
        freq = 1.0 / dt
        
        # Calculate derivative
        dx = (x - self.x_prev) * freq
        
        # Filter derivative
        edx = self._lowpass_filter(dx, self.dx_prev, self.alpha(freq, self.dcutoff))
        
        # Calculate cutoff frequency
        cutoff = self.mincutoff + self.beta * abs(edx)
        
        # Filter signal
        ex = self._lowpass_filter(x, self.x_prev, self.alpha(freq, cutoff))
        
        # Update state
        self.x_prev = ex
        self.dx_prev = edx
        self.t_prev = t
        
        return ex

    def alpha(self, freq: float, cutoff: float) -> float:
        """Calculate smoothing factor."""
        tau = 1.0 / (2 * np.pi * cutoff)
        te = 1.0 / freq
        return 1.0 / (1.0 + tau / te)

    def _lowpass_filter(self, x: float, x_prev: float, alpha: float) -> float:
        """Apply low-pass filter."""
        return alpha * x + (1.0 - alpha) * x_prev

class KalmanFilter:
    """Kalman filter for gesture smoothing."""
    
    def __init__(self, process_variance: float = 1e-3, 
                 measurement_variance: float = 1e-1):
        """
        Initialize Kalman filter.
        
        Args:
            process_variance: Process noise variance
            measurement_variance: Measurement noise variance
        """
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance
        
        self.posteri_estimate = 0.0
        self.posteri_error_estimate = 1.0

    def update(self, measurement: float) -> float:
        """Update filter with new measurement."""
        # Prediction
        priori_estimate = self.posteri_estimate
        priori_error_estimate = self.posteri_error_estimate + self.process_variance
        
        # Update
        blending_factor = priori_error_estimate / (priori_error_estimate + self.measurement_variance)
        self.posteri_estimate = priori_estimate + blending_factor * (measurement - priori_estimate)
        self.posteri_error_estimate = (1 - blending_factor) * priori_error_estimate
        
        return self.posteri_estimate

class GestureCalibrationFlow:
    """Interactive calibration flow for personalized gesture recognition."""
    
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        
        # Calibration stages
        self.calibration_stages = [
            {"name": "hand_size", "description": "Measure hand size", "samples_needed": 30},
            {"name": "finger_flexibility", "description": "Test finger flexibility", "samples_needed": 50},
            {"name": "gesture_thresholds", "description": "Calibrate gesture thresholds", "samples_needed": 20},
            {"name": "lighting_adaptation", "description": "Test different lighting", "samples_needed": 40}
        ]
        
        self.current_stage = 0
        self.stage_samples = []
        self.calibration_data = HandCalibrationData(
            hand_size=0.0,
            finger_lengths={},
            palm_size=0.0,
            joint_flexibility={},
            gesture_thresholds={},
            lighting_adaptation={},
            calibration_date=time.strftime("%Y-%m-%d %H:%M:%S")
        )

    def start_calibration(self, user_id: str = "default") -> Dict[str, Any]:
        """Start the calibration process."""
        self.calibration_data.user_id = user_id
        self.current_stage = 0
        self.stage_samples = []
        
        return {
            "status": "started",
            "current_stage": self.calibration_stages[0],
            "total_stages": len(self.calibration_stages),
            "instructions": "Place your hand flat in front of the camera with fingers spread"
        }

    def process_calibration_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process a frame during calibration."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        if not results.multi_hand_landmarks:
            return {
                "status": "no_hand_detected",
                "message": "Please place your hand in view of the camera"
            }
        
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
        
        # Process based on current stage
        stage = self.calibration_stages[self.current_stage]
        stage_name = stage["name"]
        
        if stage_name == "hand_size":
            return self._calibrate_hand_size(landmarks, frame)
        elif stage_name == "finger_flexibility":
            return self._calibrate_finger_flexibility(landmarks, frame)
        elif stage_name == "gesture_thresholds":
            return self._calibrate_gesture_thresholds(landmarks, frame)
        elif stage_name == "lighting_adaptation":
            return self._calibrate_lighting_adaptation(landmarks, frame)
        
        return {"status": "error", "message": "Unknown calibration stage"}

    def _calibrate_hand_size(self, landmarks: List[Tuple[float, float, float]], 
                           frame: np.ndarray) -> Dict[str, Any]:
        """Calibrate hand size measurements."""
        # Calculate hand span (thumb tip to pinky tip)
        thumb_tip = landmarks[4]
        pinky_tip = landmarks[20]
        hand_span = euclidean([thumb_tip[0], thumb_tip[1]], [pinky_tip[0], pinky_tip[1]])
        
        # Calculate palm size (wrist to middle finger base)
        wrist = landmarks[0]
        middle_base = landmarks[9]
        palm_length = euclidean([wrist[0], wrist[1]], [middle_base[0], middle_base[1]])
        
        # Calculate finger lengths
        finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        finger_bases = [2, 5, 9, 13, 17]
        finger_names = ["thumb", "index", "middle", "ring", "pinky"]
        
        finger_lengths = {}
        for i, (tip, base, name) in enumerate(zip(finger_tips, finger_bases, finger_names)):
            length = euclidean([landmarks[tip][0], landmarks[tip][1]], 
                             [landmarks[base][0], landmarks[base][1]])
            finger_lengths[name] = length
        
        # Store sample
        sample = {
            "hand_span": hand_span,
            "palm_length": palm_length,
            "finger_lengths": finger_lengths
        }
        self.stage_samples.append(sample)
        
        # Check if enough samples
        samples_needed = self.calibration_stages[self.current_stage]["samples_needed"]
        if len(self.stage_samples) >= samples_needed:
            # Calculate averages
            self.calibration_data.hand_size = statistics.mean([s["hand_span"] for s in self.stage_samples])
            self.calibration_data.palm_size = statistics.mean([s["palm_length"] for s in self.stage_samples])
            
            for finger in finger_names:
                lengths = [s["finger_lengths"][finger] for s in self.stage_samples]
                self.calibration_data.finger_lengths[finger] = statistics.mean(lengths)
            
            return self._advance_stage()
        
        # Draw guidance
        frame_with_guidance = self._draw_hand_size_guidance(frame, landmarks)
        
        return {
            "status": "collecting",
            "progress": len(self.stage_samples) / samples_needed,
            "message": f"Collecting hand size data: {len(self.stage_samples)}/{samples_needed}",
            "frame": frame_with_guidance
        }

    def _calibrate_finger_flexibility(self, landmarks: List[Tuple[float, float, float]], 
                                    frame: np.ndarray) -> Dict[str, Any]:
        """Calibrate finger flexibility ranges."""
        # Calculate joint angles for flexibility assessment
        finger_joints = {
            "thumb": [(1, 2, 3), (2, 3, 4)],
            "index": [(5, 6, 7), (6, 7, 8)],
            "middle": [(9, 10, 11), (10, 11, 12)],
            "ring": [(13, 14, 15), (14, 15, 16)],
            "pinky": [(17, 18, 19), (18, 19, 20)]
        }
        
        joint_angles = {}
        for finger, joints in finger_joints.items():
            finger_angles = []
            for joint in joints:
                p1 = np.array([landmarks[joint[0]][0], landmarks[joint[0]][1]])
                p2 = np.array([landmarks[joint[1]][0], landmarks[joint[1]][1]])
                p3 = np.array([landmarks[joint[2]][0], landmarks[joint[2]][1]])
                
                # Calculate angle
                v1 = p1 - p2
                v2 = p3 - p2
                angle = np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1.0, 1.0))
                finger_angles.append(np.degrees(angle))
            
            joint_angles[finger] = finger_angles
        
        self.stage_samples.append(joint_angles)
        
        samples_needed = self.calibration_stages[self.current_stage]["samples_needed"]
        if len(self.stage_samples) >= samples_needed:
            # Calculate flexibility ranges
            flexibility_ranges = {}
            for finger in joint_angles.keys():
                finger_samples = [s[finger] for s in self.stage_samples]
                # Calculate min/max for each joint
                joint_ranges = []
                for joint_idx in range(len(finger_samples[0])):
                    joint_values = [sample[joint_idx] for sample in finger_samples]
                    joint_ranges.append({
                        "min": min(joint_values),
                        "max": max(joint_values),
                        "range": max(joint_values) - min(joint_values)
                    })
                flexibility_ranges[finger] = joint_ranges
            
            self.calibration_data.joint_flexibility = flexibility_ranges
            return self._advance_stage()
        
        frame_with_guidance = self._draw_flexibility_guidance(frame, landmarks)
        
        return {
            "status": "collecting",
            "progress": len(self.stage_samples) / samples_needed,
            "message": f"Move your fingers through full range of motion: {len(self.stage_samples)}/{samples_needed}",
            "frame": frame_with_guidance
        }

    def _calibrate_gesture_thresholds(self, landmarks: List[Tuple[float, float, float]], 
                                    frame: np.ndarray) -> Dict[str, Any]:
        """Calibrate gesture-specific thresholds."""
        # Extract features for threshold calibration
        gesture_features = self.feature_extractor.extract_features(landmarks)
        
        # Convert GestureFeatures to a flat array of numerical values
        features_array = self._extract_numerical_features(gesture_features)
        
        # Store features for threshold calculation
        self.stage_samples.append(features_array)
        
        samples_needed = self.calibration_stages[self.current_stage]["samples_needed"]
        if len(self.stage_samples) >= samples_needed:
            # Calculate adaptive thresholds based on user's natural hand position
            feature_means = np.mean(self.stage_samples, axis=0)
            feature_stds = np.std(self.stage_samples, axis=0)
            
            # Define gesture-specific thresholds as multiples of standard deviation
            # Use safe indexing to avoid out-of-bounds errors
            thresholds = {
                "finger_curl_sensitivity": feature_stds[0] * 2.0 if len(feature_stds) > 0 else 0.1,
                "hand_movement_sensitivity": feature_stds[min(10, len(feature_stds)-1)] * 1.5 if len(feature_stds) > 10 else 0.1,
                "gesture_confidence_min": 0.6,  # Minimum confidence for gesture detection
                "stability_threshold": feature_stds[min(20, len(feature_stds)-1)] * 0.8 if len(feature_stds) > 20 else 0.1
            }
            
            self.calibration_data.gesture_thresholds = thresholds
            return self._advance_stage()
        
        return {
            "status": "collecting",
            "progress": len(self.stage_samples) / samples_needed,
            "message": f"Hold your hand in a relaxed position: {len(self.stage_samples)}/{samples_needed}",
            "frame": frame
        }

    def _extract_numerical_features(self, gesture_features) -> List[float]:
        """Extract numerical features from GestureFeatures object."""
        features = []
        
        # Add finger angles
        if hasattr(gesture_features, 'finger_angles') and gesture_features.finger_angles:
            features.extend(gesture_features.finger_angles)
        
        # Add finger distances
        if hasattr(gesture_features, 'finger_distances') and gesture_features.finger_distances:
            features.extend(gesture_features.finger_distances)
        
        # Add inter-finger angles
        if hasattr(gesture_features, 'inter_finger_angles') and gesture_features.inter_finger_angles:
            features.extend(gesture_features.inter_finger_angles)
        
        # Add hand size
        if hasattr(gesture_features, 'hand_size'):
            features.append(gesture_features.hand_size)
        
        # Add palm center coordinates
        if hasattr(gesture_features, 'palm_center') and gesture_features.palm_center:
            features.extend(gesture_features.palm_center)
        
        # Add palm normal vector
        if hasattr(gesture_features, 'palm_normal') and gesture_features.palm_normal:
            features.extend(gesture_features.palm_normal)
        
        # Add fingertip distances
        if hasattr(gesture_features, 'fingertip_distances') and gesture_features.fingertip_distances:
            features.extend(gesture_features.fingertip_distances)
        
        # Add thumb-finger distances
        if hasattr(gesture_features, 'thumb_finger_distances') and gesture_features.thumb_finger_distances:
            features.extend(gesture_features.thumb_finger_distances)
        
        # Add confidence scores
        if hasattr(gesture_features, 'detection_confidence'):
            features.append(gesture_features.detection_confidence)
        if hasattr(gesture_features, 'tracking_confidence'):
            features.append(gesture_features.tracking_confidence)
        
        # If we still don't have enough features, flatten the landmarks
        if len(features) < 10 and hasattr(gesture_features, 'landmarks') and gesture_features.landmarks:
            for landmark in gesture_features.landmarks:
                if isinstance(landmark, (list, tuple)):
                    features.extend(landmark[:2])  # Only x, y to avoid too many features
        
        return features

    def _calibrate_lighting_adaptation(self, landmarks: List[Tuple[float, float, float]], 
                                     frame: np.ndarray) -> Dict[str, Any]:
        """Calibrate for different lighting conditions."""
        # Analyze lighting conditions
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate lighting metrics
        brightness = np.mean(gray_frame)
        contrast = np.std(gray_frame)
        
        # Extract features under current lighting
        gesture_features = self.feature_extractor.extract_features(landmarks)
        features_array = self._extract_numerical_features(gesture_features)
        
        lighting_sample = {
            "brightness": brightness,
            "contrast": contrast,
            "features": features_array
        }
        self.stage_samples.append(lighting_sample)
        
        samples_needed = self.calibration_stages[self.current_stage]["samples_needed"]
        if len(self.stage_samples) >= samples_needed:
            # Calculate lighting adaptation parameters
            brightness_values = [s["brightness"] for s in self.stage_samples]
            contrast_values = [s["contrast"] for s in self.stage_samples]
            
            adaptation_params = {
                "brightness_range": {"min": min(brightness_values), "max": max(brightness_values)},
                "contrast_range": {"min": min(contrast_values), "max": max(contrast_values)},
                "feature_scaling": self._calculate_lighting_feature_scaling()
            }
            
            self.calibration_data.lighting_adaptation = adaptation_params
            return self._finish_calibration()
        
        return {
            "status": "collecting",
            "progress": len(self.stage_samples) / samples_needed,
            "message": f"Testing different lighting conditions: {len(self.stage_samples)}/{samples_needed}",
            "instruction": "Move to different lighting or adjust room lighting",
            "frame": frame
        }

    def _calculate_lighting_feature_scaling(self) -> Dict[str, float]:
        """Calculate feature scaling factors for different lighting conditions."""
        # Group samples by lighting conditions
        bright_samples = [s for s in self.stage_samples if s["brightness"] > 150]
        dim_samples = [s for s in self.stage_samples if s["brightness"] < 100]
        
        if not bright_samples or not dim_samples:
            return {"bright_scale": 1.0, "dim_scale": 1.0}
        
        # Calculate scaling factors
        bright_features = np.mean([s["features"] for s in bright_samples], axis=0)
        dim_features = np.mean([s["features"] for s in dim_samples], axis=0)
        
        # Avoid division by zero
        scaling_factors = {}
        for i in range(len(bright_features)):
            if dim_features[i] != 0:
                scaling_factors[f"feature_{i}"] = bright_features[i] / dim_features[i]
            else:
                scaling_factors[f"feature_{i}"] = 1.0
        
        return scaling_factors

    def _advance_stage(self) -> Dict[str, Any]:
        """Advance to the next calibration stage."""
        self.current_stage += 1
        self.stage_samples = []
        
        if self.current_stage >= len(self.calibration_stages):
            return self._finish_calibration()
        
        next_stage = self.calibration_stages[self.current_stage]
        return {
            "status": "stage_complete",
            "next_stage": next_stage,
            "progress": self.current_stage / len(self.calibration_stages),
            "message": f"Moving to: {next_stage['description']}"
        }

    def _finish_calibration(self) -> Dict[str, Any]:
        """Finish the calibration process."""
        self.calibration_data.num_calibration_samples = sum(
            stage["samples_needed"] for stage in self.calibration_stages
        )
        
        return {
            "status": "complete",
            "calibration_data": asdict(self.calibration_data),
            "message": "Calibration completed successfully!"
        }

    def _draw_hand_size_guidance(self, frame: np.ndarray, landmarks: List[Tuple[float, float, float]]) -> np.ndarray:
        """Draw guidance for hand size calibration."""
        h, w = frame.shape[:2]
        
        # Draw hand outline
        thumb_tip = (int(landmarks[4][0] * w), int(landmarks[4][1] * h))
        pinky_tip = (int(landmarks[20][0] * w), int(landmarks[20][1] * h))
        
        cv2.line(frame, thumb_tip, pinky_tip, (0, 255, 0), 2)
        cv2.putText(frame, "Keep hand flat with fingers spread", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame

    def _draw_flexibility_guidance(self, frame: np.ndarray, landmarks: List[Tuple[float, float, float]]) -> np.ndarray:
        """Draw guidance for flexibility calibration."""
        cv2.putText(frame, "Move fingers through full range of motion", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, "Make fists, point, spread fingers", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return frame

    def save_calibration(self, filepath: str):
        """Save calibration data to file."""
        with open(filepath, 'w') as f:
            json.dump(asdict(self.calibration_data), f, indent=2)

    def load_calibration(self, filepath: str) -> HandCalibrationData:
        """Load calibration data from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.calibration_data = HandCalibrationData(**data)
        return self.calibration_data

class AdaptiveGestureFilter:
    """Adaptive filtering system for gesture recognition."""
    
    def __init__(self, calibration_data: Optional[HandCalibrationData] = None):
        self.calibration_data = calibration_data
        
        # Initialize filters
        self.one_euro_filters = {}
        self.kalman_filters = {}
        
        # Confidence tracking
        self.confidence_history = deque(maxlen=30)
        self.gesture_stability = deque(maxlen=10)
        
        # Adaptive parameters
        self.adaptive_threshold = 0.7
        self.stability_requirement = 0.8

    def initialize_filters(self, feature_names: List[str]):
        """Initialize filters for each feature."""
        for name in feature_names:
            self.one_euro_filters[name] = OneEuroFilter(
                freq=30.0,
                mincutoff=1.0,
                beta=0.007
            )
            self.kalman_filters[name] = KalmanFilter(
                process_variance=1e-3,
                measurement_variance=1e-1
            )

    def filter_features(self, features: Dict[str, float], timestamp: Optional[float] = None) -> Dict[str, float]:
        """Apply filtering to gesture features."""
        filtered_features = {}
        
        for name, value in features.items():
            if name not in self.one_euro_filters:
                self.one_euro_filters[name] = OneEuroFilter()
                self.kalman_filters[name] = KalmanFilter()
            
            # Apply One Euro filter
            filtered_value = self.one_euro_filters[name](value, timestamp)
            
            # Apply Kalman filter
            filtered_value = self.kalman_filters[name].update(filtered_value)
            
            filtered_features[name] = filtered_value
        
        return filtered_features

    def apply_calibration_scaling(self, features: Dict[str, float], 
                                lighting_conditions: Dict[str, float]) -> Dict[str, float]:
        """Apply calibration-based feature scaling."""
        if not self.calibration_data or not self.calibration_data.lighting_adaptation:
            return features
        
        scaling = self.calibration_data.lighting_adaptation.get("feature_scaling", {})
        brightness = lighting_conditions.get("brightness", 128)
        
        # Determine lighting condition
        if brightness > 150:
            scale_factor = 1.0  # Bright conditions (reference)
        elif brightness < 100:
            scale_factor = 0.9  # Dim conditions
        else:
            # Linear interpolation
            scale_factor = 0.9 + 0.1 * (brightness - 100) / 50
        
        scaled_features = {}
        for name, value in features.items():
            if name in scaling:
                scaled_features[name] = value * scaling[name] * scale_factor
            else:
                scaled_features[name] = value
        
        return scaled_features

    def update_confidence_tracking(self, gesture: str, confidence: float):
        """Update confidence tracking for adaptive thresholding."""
        self.confidence_history.append(confidence)
        self.gesture_stability.append(gesture)
        
        # Adaptive threshold adjustment
        if len(self.confidence_history) >= 10:
            avg_confidence = statistics.mean(list(self.confidence_history)[-10:])
            if avg_confidence > 0.9:
                self.adaptive_threshold = max(0.6, self.adaptive_threshold - 0.01)
            elif avg_confidence < 0.6:
                self.adaptive_threshold = min(0.8, self.adaptive_threshold + 0.01)

    def is_gesture_stable(self, current_gesture: str) -> bool:
        """Check if gesture is stable enough for action."""
        if len(self.gesture_stability) < 5:
            return False
        
        recent_gestures = list(self.gesture_stability)[-5:]
        stability_ratio = recent_gestures.count(current_gesture) / len(recent_gestures)
        
        return stability_ratio >= self.stability_requirement

    def get_adaptive_threshold(self) -> float:
        """Get current adaptive threshold."""
        return self.adaptive_threshold

    def reset_tracking(self):
        """Reset tracking history."""
        self.confidence_history.clear()
        self.gesture_stability.clear()
        self.adaptive_threshold = 0.7
