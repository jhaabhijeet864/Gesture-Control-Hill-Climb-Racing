#!/usr/bin/env python3
"""
Integrated ML Gesture Recognition System
Combines traditional ML, temporal models, calibration, and filtering
"""

import numpy as np
import cv2
import mediapipe as mp
import time
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from collections import deque
import json
import os
import threading
from queue import Queue
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from ml_gesture_recognition import FeatureExtractor, MLGestureClassifier
from temporal_gesture_models import TemporalGestureRecognizer, TemporalGestureSequence
from calibration_filtering import GestureCalibrationFlow, AdaptiveGestureFilter, HandCalibrationData

@dataclass
class GestureResult:
    """Result of gesture recognition."""
    gesture: str
    confidence: float
    method: str  # 'ml', 'temporal', 'hybrid'
    timestamp: float
    features: Dict[str, float]
    filtered_features: Dict[str, float]
    is_stable: bool

class RobustGestureRecognitionSystem:
    """
    Integrated gesture recognition system combining:
    - Traditional ML classifiers (SVM, Random Forest)
    - Temporal models (LSTM, GRU, TCN)
    - Calibration system
    - Adaptive filtering
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._default_config()
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=self.config.get('detection_confidence', 0.7),
            min_tracking_confidence=self.config.get('tracking_confidence', 0.5)
        )
        
        # Initialize components
        self.feature_extractor = FeatureExtractor()
        self.ml_classifier = MLGestureClassifier()
        self.temporal_recognizer = TemporalGestureRecognizer(
            model_type=self.config.get('temporal_model', 'lstm')
        )
        self.calibration_flow = GestureCalibrationFlow(self.feature_extractor)
        self.adaptive_filter = AdaptiveGestureFilter()
        
        # System state
        self.is_calibrated = False
        self.is_ml_trained = False
        self.is_temporal_trained = False
        self.calibration_data = None
        
        # Real-time processing
        self.frame_queue = Queue(maxsize=10)
        self.result_queue = Queue(maxsize=5)
        self.processing_thread = None
        self.is_running = False
        
        # Performance tracking
        self.performance_metrics = {
            'frames_processed': 0,
            'gestures_detected': 0,
            'average_confidence': 0.0,
            'processing_fps': 0.0,
            'last_fps_update': time.time()
        }
        
        # Feature and result buffers
        self.feature_buffer = deque(maxlen=50)
        self.result_history = deque(maxlen=100)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration."""
        return {
            'detection_confidence': 0.7,
            'tracking_confidence': 0.5,
            'temporal_model': 'lstm',
            'ml_algorithm': 'random_forest',
            'hybrid_mode': True,
            'confidence_threshold': 0.6,
            'stability_frames': 5,
            'max_sequence_length': 50,
            'use_filtering': True,
            'calibration_required': True
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

    def start_calibration(self, user_id: str = "default") -> Dict[str, Any]:
        """Start user calibration process."""
        return self.calibration_flow.start_calibration(user_id)

    def process_calibration_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process frame during calibration."""
        result = self.calibration_flow.process_calibration_frame(frame)
        
        if result.get('status') == 'complete':
            self.calibration_data = HandCalibrationData(**result['calibration_data'])
            self.adaptive_filter = AdaptiveGestureFilter(self.calibration_data)
            self.is_calibrated = True
            
            # Save calibration
            self._save_calibration()
        
        return result

    def load_calibration(self, filepath: str = None) -> bool:
        """Load existing calibration data."""
        try:
            if filepath is None:
                filepath = "calibration_data.json"
            
            self.calibration_data = self.calibration_flow.load_calibration(filepath)
            self.adaptive_filter = AdaptiveGestureFilter(self.calibration_data)
            self.is_calibrated = True
            
            self.logger.info(f"Loaded calibration for user: {self.calibration_data.user_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load calibration: {e}")
            return False

    def _save_calibration(self, filepath: str = None):
        """Save calibration data."""
        if filepath is None:
            filepath = "calibration_data.json"
        
        try:
            self.calibration_flow.save_calibration(filepath)
            self.logger.info(f"Saved calibration to: {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save calibration: {e}")

    def train_ml_classifier(self, training_data: List[Tuple[List[float], str]]) -> Dict[str, Any]:
        """Train the ML classifier."""
        try:
            # Prepare training data
            features = [sample[0] for sample in training_data]
            labels = [sample[1] for sample in training_data]
            
            # Train with cross-validation
            results = self.ml_classifier.train_with_cross_validation(
                features, labels, 
                algorithm=self.config.get('ml_algorithm', 'random_forest')
            )
            
            self.is_ml_trained = True
            self.logger.info(f"ML classifier trained with accuracy: {results['accuracy']:.3f}")
            
            return results
        except Exception as e:
            self.logger.error(f"Failed to train ML classifier: {e}")
            return {"success": False, "error": str(e)}

    def train_temporal_model(self, sequences: List[TemporalGestureSequence]) -> Dict[str, Any]:
        """Train the temporal model."""
        try:
            results = self.temporal_recognizer.train(sequences)
            self.is_temporal_trained = True
            
            self.logger.info(f"Temporal model trained with accuracy: {results['final_val_accuracy']:.3f}")
            return results
        except Exception as e:
            self.logger.error(f"Failed to train temporal model: {e}")
            return {"success": False, "error": str(e)}

    def process_frame(self, frame: np.ndarray) -> Optional[GestureResult]:
        """Process a single frame and return gesture result."""
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            if not results.multi_hand_landmarks:
                return None
            
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
            
            # Extract features
            gesture_features = self.feature_extractor.extract_features(landmarks)
            features = self._extract_numerical_features(gesture_features)
            timestamp = time.time()
            
            # Apply filtering if enabled
            if self.config.get('use_filtering', True) and self.is_calibrated:
                # Get lighting conditions
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                lighting_conditions = {
                    'brightness': np.mean(gray_frame),
                    'contrast': np.std(gray_frame)
                }
                
                # Apply calibration scaling
                scaled_features = self.adaptive_filter.apply_calibration_scaling(
                    {f"f_{i}": v for i, v in enumerate(features)}, 
                    lighting_conditions
                )
                
                # Apply temporal filtering
                filtered_features = self.adaptive_filter.filter_features(
                    scaled_features, timestamp
                )
                
                # Convert back to list
                features_filtered = [filtered_features[f"f_{i}"] for i in range(len(features))]
            else:
                features_filtered = features
                filtered_features = {f"f_{i}": v for i, v in enumerate(features_filtered)}
            
            # Store in buffer for temporal analysis
            self.feature_buffer.append(features_filtered)
            
            # Gesture recognition
            gesture_result = self._recognize_gesture(features, features_filtered, timestamp)
            
            # Update performance metrics
            self._update_performance_metrics()
            
            return gesture_result
            
        except Exception as e:
            self.logger.error(f"Error processing frame: {e}")
            return None

    def _recognize_gesture(self, original_features: List[float], 
                          filtered_features: List[float], timestamp: float) -> GestureResult:
        """Perform gesture recognition using available methods."""
        
        gesture_candidates = {}
        confidence_scores = {}
        
        # Method 1: ML Classifier
        if self.is_ml_trained:
            try:
                ml_result = self.ml_classifier.predict(filtered_features)
                gesture_candidates['ml'] = ml_result['predicted_class']
                confidence_scores['ml'] = ml_result['confidence']
            except Exception as e:
                self.logger.warning(f"ML prediction failed: {e}")
        
        # Method 2: Temporal Model
        if self.is_temporal_trained and len(self.feature_buffer) >= 10:
            try:
                temporal_gesture, temporal_confidence = self.temporal_recognizer.predict_realtime(
                    filtered_features, timestamp
                )
                gesture_candidates['temporal'] = temporal_gesture
                confidence_scores['temporal'] = temporal_confidence
            except Exception as e:
                self.logger.warning(f"Temporal prediction failed: {e}")
        
        # Hybrid decision making
        if self.config.get('hybrid_mode', True) and len(gesture_candidates) > 1:
            final_gesture, final_confidence, method = self._hybrid_decision(
                gesture_candidates, confidence_scores
            )
        elif gesture_candidates:
            # Use best single method
            best_method = max(confidence_scores.keys(), key=lambda k: confidence_scores[k])
            final_gesture = gesture_candidates[best_method]
            final_confidence = confidence_scores[best_method]
            method = best_method
        else:
            # Fallback
            final_gesture = "unknown"
            final_confidence = 0.0
            method = "fallback"
        
        # Check stability
        is_stable = True
        if self.is_calibrated:
            self.adaptive_filter.update_confidence_tracking(final_gesture, final_confidence)
            is_stable = self.adaptive_filter.is_gesture_stable(final_gesture)
        
        # Create result
        result = GestureResult(
            gesture=final_gesture,
            confidence=final_confidence,
            method=method,
            timestamp=timestamp,
            features={f"f_{i}": v for i, v in enumerate(original_features)},
            filtered_features={f"f_{i}": v for i, v in enumerate(filtered_features)},
            is_stable=is_stable
        )
        
        # Store in history
        self.result_history.append(result)
        
        return result

    def _hybrid_decision(self, candidates: Dict[str, str], 
                        confidences: Dict[str, float]) -> Tuple[str, float, str]:
        """Make hybrid decision from multiple methods."""
        
        # Weight different methods
        method_weights = {
            'ml': 0.6,
            'temporal': 0.8,  # Temporal models often more robust
            'rule_based': 0.3
        }
        
        # Calculate weighted scores for each gesture
        gesture_scores = {}
        
        for method, gesture in candidates.items():
            confidence = confidences[method]
            weight = method_weights.get(method, 0.5)
            weighted_score = confidence * weight
            
            if gesture not in gesture_scores:
                gesture_scores[gesture] = []
            gesture_scores[gesture].append(weighted_score)
        
        # Find best gesture
        best_gesture = None
        best_score = 0.0
        
        for gesture, scores in gesture_scores.items():
            # Use maximum score (best method for this gesture)
            gesture_score = max(scores)
            if gesture_score > best_score:
                best_score = gesture_score
                best_gesture = gesture
        
        # Determine which method contributed most
        contributing_method = "hybrid"
        for method, gesture in candidates.items():
            if gesture == best_gesture and confidences[method] * method_weights.get(method, 0.5) == best_score:
                contributing_method = method
                break
        
        return best_gesture, best_score, contributing_method

    def start_realtime_processing(self):
        """Start real-time gesture processing in separate thread."""
        if self.is_running:
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        self.logger.info("Started real-time gesture processing")

    def stop_realtime_processing(self):
        """Stop real-time processing."""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join()
        
        self.logger.info("Stopped real-time gesture processing")

    def add_frame(self, frame: np.ndarray) -> bool:
        """Add frame to processing queue."""
        try:
            self.frame_queue.put_nowait(frame)
            return True
        except:
            return False

    def get_latest_result(self) -> Optional[GestureResult]:
        """Get latest gesture recognition result."""
        try:
            return self.result_queue.get_nowait()
        except:
            return None

    def _processing_loop(self):
        """Main processing loop for real-time recognition."""
        while self.is_running:
            try:
                # Get frame from queue
                frame = self.frame_queue.get(timeout=0.1)
                
                # Process frame
                result = self.process_frame(frame)
                
                if result:
                    # Add to result queue
                    try:
                        self.result_queue.put_nowait(result)
                    except:
                        # Queue full, remove oldest
                        try:
                            self.result_queue.get_nowait()
                            self.result_queue.put_nowait(result)
                        except:
                            pass
                
            except Exception as e:
                if self.is_running:  # Only log if not shutting down
                    continue

    def _update_performance_metrics(self):
        """Update performance tracking metrics."""
        self.performance_metrics['frames_processed'] += 1
        
        current_time = time.time()
        time_elapsed = current_time - self.performance_metrics['last_fps_update']
        
        if time_elapsed >= 1.0:  # Update every second
            fps = self.performance_metrics['frames_processed'] / time_elapsed
            self.performance_metrics['processing_fps'] = fps
            self.performance_metrics['frames_processed'] = 0
            self.performance_metrics['last_fps_update'] = current_time
            
            # Update average confidence
            if self.result_history:
                recent_results = list(self.result_history)[-30:]  # Last 30 results
                avg_confidence = np.mean([r.confidence for r in recent_results])
                self.performance_metrics['average_confidence'] = avg_confidence

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics."""
        return self.performance_metrics.copy()

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "calibrated": self.is_calibrated,
            "ml_trained": self.is_ml_trained,
            "temporal_trained": self.is_temporal_trained,
            "realtime_processing": self.is_running,
            "calibration_user": self.calibration_data.user_id if self.calibration_data else None,
            "feature_buffer_size": len(self.feature_buffer),
            "result_history_size": len(self.result_history),
            "performance": self.get_performance_metrics(),
            "config": self.config
        }

    def save_models(self, directory: str = "models"):
        """Save all trained models."""
        os.makedirs(directory, exist_ok=True)
        
        try:
            if self.is_ml_trained:
                self.ml_classifier.save_model(os.path.join(directory, "ml_classifier.pkl"))
            
            if self.is_temporal_trained:
                self.temporal_recognizer.save_model(os.path.join(directory, "temporal_model.pth"))
            
            if self.is_calibrated:
                self._save_calibration(os.path.join(directory, "calibration_data.json"))
            
            # Save config
            with open(os.path.join(directory, "system_config.json"), 'w') as f:
                json.dump(self.config, f, indent=2)
            
            self.logger.info(f"Models saved to: {directory}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save models: {e}")
            return False

    def load_models(self, directory: str = "models") -> Dict[str, bool]:
        """Load all available models."""
        results = {
            "ml_classifier": False,
            "temporal_model": False,
            "calibration": False,
            "config": False
        }
        
        try:
            # Load ML classifier
            ml_path = os.path.join(directory, "ml_classifier.pkl")
            if os.path.exists(ml_path):
                self.ml_classifier.load_model(ml_path)
                self.is_ml_trained = True
                results["ml_classifier"] = True
            
            # Load temporal model
            temporal_path = os.path.join(directory, "temporal_model.pth")
            if os.path.exists(temporal_path):
                self.temporal_recognizer.load_model(temporal_path)
                self.is_temporal_trained = True
                results["temporal_model"] = True
            
            # Load calibration
            calib_path = os.path.join(directory, "calibration_data.json")
            if os.path.exists(calib_path):
                results["calibration"] = self.load_calibration(calib_path)
            
            # Load config
            config_path = os.path.join(directory, "system_config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.config.update(json.load(f))
                results["config"] = True
            
            self.logger.info(f"Loaded models from: {directory}")
            
        except Exception as e:
            self.logger.error(f"Failed to load models: {e}")
        
        return results

    def cleanup(self):
        """Cleanup resources."""
        self.stop_realtime_processing()
        if hasattr(self, 'hands'):
            self.hands.close()
        
        self.logger.info("System cleanup completed")
