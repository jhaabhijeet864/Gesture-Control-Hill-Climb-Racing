#!/usr/bin/env python3
"""
ML Gesture Recognition - Robust machine learning-based gesture classification
"""

import numpy as np
import pandas as pd
import pickle
import json
import os
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib
import math

@dataclass
class GestureFeatures:
    """Engineered features from MediaPipe hand landmarks."""
    
    # Basic landmark positions (normalized)
    landmarks: List[List[float]]  # 21 landmarks x [x, y, z]
    
    # Geometric features
    finger_angles: List[float]    # 5 finger bend angles
    finger_distances: List[float] # Distances from palm center
    inter_finger_angles: List[float] # Angles between adjacent fingers
    
    # Hand properties
    hand_size: float             # Diagonal span of hand
    palm_center: List[float]     # Center of palm
    palm_normal: List[float]     # Palm normal vector
    
    # Relative positions
    fingertip_distances: List[float]  # Distances between fingertips
    thumb_finger_distances: List[float] # Thumb to other fingertips
    
    # Temporal features (for sequence-based models)
    velocity: Optional[List[float]] = None    # Landmark velocities
    acceleration: Optional[List[float]] = None # Landmark accelerations
    
    # Confidence scores
    detection_confidence: float = 1.0
    tracking_confidence: float = 1.0

class FeatureExtractor:
    """Extract engineered features from MediaPipe hand landmarks."""
    
    def __init__(self):
        # MediaPipe hand landmark indices
        self.WRIST = 0
        self.THUMB_TIP = 4
        self.INDEX_TIP = 8
        self.MIDDLE_TIP = 12
        self.RING_TIP = 16
        self.PINKY_TIP = 20
        
        # Finger landmark ranges
        self.FINGER_LANDMARKS = {
            'thumb': [1, 2, 3, 4],
            'index': [5, 6, 7, 8],
            'middle': [9, 10, 11, 12],
            'ring': [13, 14, 15, 16],
            'pinky': [17, 18, 19, 20]
        }
        
        # Previous landmarks for velocity/acceleration
        self.prev_landmarks = None
        self.prev_velocity = None
        self.prev_timestamp = None

    def extract_features(self, landmarks: List[List[float]], 
                        hand_type: str = "Right", 
                        timestamp: float = None) -> GestureFeatures:
        """Extract comprehensive features from hand landmarks."""
        
        if len(landmarks) != 21:
            raise ValueError(f"Expected 21 landmarks, got {len(landmarks)}")
        
        landmarks = np.array(landmarks)
        
        # Basic geometric features
        finger_angles = self._calculate_finger_angles(landmarks)
        finger_distances = self._calculate_finger_distances(landmarks)
        inter_finger_angles = self._calculate_inter_finger_angles(landmarks)
        
        # Hand properties
        hand_size = self._calculate_hand_size(landmarks)
        palm_center = self._calculate_palm_center(landmarks)
        palm_normal = self._calculate_palm_normal(landmarks)
        
        # Relative distances
        fingertip_distances = self._calculate_fingertip_distances(landmarks)
        thumb_finger_distances = self._calculate_thumb_finger_distances(landmarks)
        
        # Temporal features
        velocity = None
        acceleration = None
        if timestamp is not None:
            velocity, acceleration = self._calculate_temporal_features(landmarks, timestamp)
        
        return GestureFeatures(
            landmarks=landmarks.tolist(),
            finger_angles=finger_angles,
            finger_distances=finger_distances,
            inter_finger_angles=inter_finger_angles,
            hand_size=hand_size,
            palm_center=palm_center.tolist(),
            palm_normal=palm_normal.tolist(),
            fingertip_distances=fingertip_distances,
            thumb_finger_distances=thumb_finger_distances,
            velocity=velocity.tolist() if velocity is not None else None,
            acceleration=acceleration.tolist() if acceleration is not None else None
        )

    def _calculate_finger_angles(self, landmarks: np.ndarray) -> List[float]:
        """Calculate bend angles for each finger."""
        angles = []
        
        for finger_name, indices in self.FINGER_LANDMARKS.items():
            if len(indices) >= 4:  # Need at least 4 points for 3 vectors
                # Get the 4 key points of the finger
                p1, p2, p3, p4 = landmarks[indices]
                
                # Calculate vectors
                v1 = p2 - p1
                v2 = p3 - p2
                v3 = p4 - p3
                
                # Calculate angles between consecutive segments
                angle1 = self._angle_between_vectors(v1, v2)
                angle2 = self._angle_between_vectors(v2, v3)
                
                # Overall finger bend (sum of segment angles)
                finger_angle = angle1 + angle2
                angles.append(finger_angle)
            else:
                angles.append(0.0)
        
        return angles

    def _calculate_finger_distances(self, landmarks: np.ndarray) -> List[float]:
        """Calculate distances from fingertips to palm center."""
        palm_center = self._calculate_palm_center(landmarks)
        
        fingertip_indices = [self.THUMB_TIP, self.INDEX_TIP, self.MIDDLE_TIP, 
                           self.RING_TIP, self.PINKY_TIP]
        
        distances = []
        for tip_idx in fingertip_indices:
            distance = np.linalg.norm(landmarks[tip_idx] - palm_center)
            distances.append(distance)
        
        return distances

    def _calculate_inter_finger_angles(self, landmarks: np.ndarray) -> List[float]:
        """Calculate angles between adjacent fingers."""
        palm_center = self._calculate_palm_center(landmarks)
        fingertip_indices = [self.THUMB_TIP, self.INDEX_TIP, self.MIDDLE_TIP, 
                           self.RING_TIP, self.PINKY_TIP]
        
        angles = []
        for i in range(len(fingertip_indices) - 1):
            # Vectors from palm center to consecutive fingertips
            v1 = landmarks[fingertip_indices[i]] - palm_center
            v2 = landmarks[fingertip_indices[i + 1]] - palm_center
            
            angle = self._angle_between_vectors(v1, v2)
            angles.append(angle)
        
        return angles

    def _calculate_hand_size(self, landmarks: np.ndarray) -> float:
        """Calculate hand size as diagonal span."""
        # Use wrist to middle fingertip as primary measure
        wrist = landmarks[self.WRIST]
        middle_tip = landmarks[self.MIDDLE_TIP]
        
        return np.linalg.norm(middle_tip - wrist)

    def _calculate_palm_center(self, landmarks: np.ndarray) -> np.ndarray:
        """Calculate center of palm using key palm landmarks."""
        # Use wrist and base knuckles
        palm_landmarks = [0, 1, 5, 9, 13, 17]  # Wrist and finger bases
        palm_points = landmarks[palm_landmarks]
        
        return np.mean(palm_points, axis=0)

    def _calculate_palm_normal(self, landmarks: np.ndarray) -> np.ndarray:
        """Calculate palm normal vector using 3 palm points."""
        # Use wrist, index base, and pinky base
        p1 = landmarks[0]   # Wrist
        p2 = landmarks[5]   # Index base
        p3 = landmarks[17]  # Pinky base
        
        # Calculate normal using cross product
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        
        # Normalize
        norm = np.linalg.norm(normal)
        if norm > 0:
            normal = normal / norm
        
        return normal

    def _calculate_fingertip_distances(self, landmarks: np.ndarray) -> List[float]:
        """Calculate distances between all fingertip pairs."""
        fingertip_indices = [self.THUMB_TIP, self.INDEX_TIP, self.MIDDLE_TIP, 
                           self.RING_TIP, self.PINKY_TIP]
        
        distances = []
        for i in range(len(fingertip_indices)):
            for j in range(i + 1, len(fingertip_indices)):
                distance = np.linalg.norm(
                    landmarks[fingertip_indices[i]] - landmarks[fingertip_indices[j]]
                )
                distances.append(distance)
        
        return distances

    def _calculate_thumb_finger_distances(self, landmarks: np.ndarray) -> List[float]:
        """Calculate distances from thumb to other fingertips."""
        thumb_tip = landmarks[self.THUMB_TIP]
        other_tips = [self.INDEX_TIP, self.MIDDLE_TIP, self.RING_TIP, self.PINKY_TIP]
        
        distances = []
        for tip_idx in other_tips:
            distance = np.linalg.norm(thumb_tip - landmarks[tip_idx])
            distances.append(distance)
        
        return distances

    def _calculate_temporal_features(self, landmarks: np.ndarray, 
                                   timestamp: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Calculate velocity and acceleration features."""
        if self.prev_landmarks is None or self.prev_timestamp is None:
            self.prev_landmarks = landmarks.copy()
            self.prev_timestamp = timestamp
            return None, None
        
        dt = timestamp - self.prev_timestamp
        if dt <= 0:
            return None, None
        
        # Calculate velocity
        velocity = (landmarks - self.prev_landmarks) / dt
        
        # Calculate acceleration
        acceleration = None
        if self.prev_velocity is not None:
            prev_dt = self.prev_timestamp - (self.prev_timestamp - dt)  # Previous dt
            if prev_dt > 0:
                acceleration = (velocity - self.prev_velocity) / dt
        
        # Update history
        self.prev_landmarks = landmarks.copy()
        self.prev_velocity = velocity.copy() if velocity is not None else None
        self.prev_timestamp = timestamp
        
        return velocity, acceleration

    def _angle_between_vectors(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate angle between two vectors in radians."""
        # Normalize vectors
        v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
        v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
        
        # Calculate angle
        cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        return angle

    def normalize_features(self, features: GestureFeatures, hand_size: float = None) -> GestureFeatures:
        """Normalize features by hand size for user-independent recognition."""
        if hand_size is None:
            hand_size = features.hand_size
        
        if hand_size <= 0:
            return features
        
        # Create a copy of features
        normalized = GestureFeatures(**asdict(features))
        
        # Normalize distance-based features
        normalized.finger_distances = [d / hand_size for d in features.finger_distances]
        normalized.fingertip_distances = [d / hand_size for d in features.fingertip_distances]
        normalized.thumb_finger_distances = [d / hand_size for d in features.thumb_finger_distances]
        normalized.hand_size = 1.0  # Normalized reference
        
        # Normalize landmark positions
        normalized.landmarks = [[x / hand_size, y / hand_size, z / hand_size] 
                              for x, y, z in features.landmarks]
        
        # Normalize velocity and acceleration if present
        if features.velocity is not None:
            normalized.velocity = [[vx / hand_size, vy / hand_size, vz / hand_size] 
                                 for vx, vy, vz in features.velocity]
        
        if features.acceleration is not None:
            normalized.acceleration = [[ax / hand_size, ay / hand_size, az / hand_size] 
                                     for ax, ay, az in features.acceleration]
        
        return normalized

class MLGestureClassifier:
    """Machine learning-based gesture classifier."""
    
    def __init__(self, model_type: str = "random_forest"):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_extractor = FeatureExtractor()
        self.is_trained = False
        
        # Model configurations
        self.model_configs = {
            "random_forest": {
                "classifier": RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42
                )
            },
            "svm": {
                "classifier": SVC(
                    kernel='rbf',
                    C=1.0,
                    gamma='scale',
                    probability=True,
                    random_state=42
                )
            }
        }

    def _features_to_vector(self, features: GestureFeatures) -> np.ndarray:
        """Convert features object to feature vector for ML."""
        vector = []
        
        # Add geometric features
        vector.extend(features.finger_angles)
        vector.extend(features.finger_distances)
        vector.extend(features.inter_finger_angles)
        vector.extend(features.fingertip_distances)
        vector.extend(features.thumb_finger_distances)
        
        # Add hand properties
        vector.append(features.hand_size)
        vector.extend(features.palm_center)
        vector.extend(features.palm_normal)
        
        # Add landmark positions (flattened)
        for landmark in features.landmarks:
            vector.extend(landmark)
        
        # Add temporal features if available
        if features.velocity is not None:
            for vel in features.velocity:
                vector.extend(vel)
        else:
            # Pad with zeros if no velocity data
            vector.extend([0.0] * (21 * 3))
        
        if features.acceleration is not None:
            for acc in features.acceleration:
                vector.extend(acc)
        else:
            # Pad with zeros if no acceleration data
            vector.extend([0.0] * (21 * 3))
        
        # Add confidence scores
        vector.append(features.detection_confidence)
        vector.append(features.tracking_confidence)
        
        return np.array(vector)

    def train(self, training_data: List[Tuple[GestureFeatures, str]], 
              validation_split: float = 0.2) -> Dict[str, Any]:
        """Train the classifier with cross-validation."""
        
        if len(training_data) == 0:
            raise ValueError("No training data provided")
        
        # Convert features to vectors
        X = []
        y = []
        
        for features, label in training_data:
            # Normalize features by hand size
            normalized_features = self.feature_extractor.normalize_features(features)
            feature_vector = self._features_to_vector(normalized_features)
            X.append(feature_vector)
            y.append(label)
        
        X = np.array(X)
        y = np.array(y)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create pipeline
        classifier = self.model_configs[self.model_type]["classifier"]
        self.model = Pipeline([
            ('classifier', classifier)
        ])
        
        # Train model
        self.model.fit(X_scaled, y_encoded)
        
        # Cross-validation
        cv_scores = self._perform_cross_validation(X_scaled, y_encoded)
        
        # Generate metrics on full dataset
        y_pred = self.model.predict(X_scaled)
        metrics = self._calculate_metrics(y_encoded, y_pred)
        
        self.is_trained = True
        
        return {
            "model_type": self.model_type,
            "training_samples": len(training_data),
            "classes": self.label_encoder.classes_.tolist(),
            "cv_scores": cv_scores,
            "metrics": metrics,
            "feature_vector_size": X.shape[1]
        }

    def train_with_cross_validation(self, features: List[List[float]], labels: List[str], 
                                   algorithm: str = "random_forest", cv_folds: int = 5) -> Dict[str, Any]:
        """Train classifier with raw feature arrays and cross-validation."""
        
        if len(features) == 0:
            raise ValueError("No training data provided")
        
        if len(features) != len(labels):
            raise ValueError("Number of features and labels must match")
        
        # Set model type
        self.model_type = algorithm
        
        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(labels)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform cross-validation
        cv_results = self._perform_cross_validation(X_scaled, y_encoded, cv_folds)
        
        # Train final model on all data
        classifier = self.model_configs[self.model_type]["classifier"]
        self.model = Pipeline([
            ('classifier', classifier)
        ])
        
        # Fit the model
        self.model.fit(X_scaled, y_encoded)
        self.is_trained = True
        
        # Calculate final metrics on training data
        y_pred = self.model.predict(X_scaled)
        final_metrics = self._calculate_metrics(y_encoded, y_pred)
        
        # Combine results
        results = {
            "model_type": self.model_type,
            "cross_validation": cv_results,
            "final_metrics": final_metrics,
            "training_samples": len(features),
            "num_classes": len(set(labels)),
            "classes": list(set(labels)),
            "accuracy": cv_results["accuracy_mean"],
            "f1_score": final_metrics["weighted_f1"]
        }
        
        return results

    def _perform_cross_validation(self, X: np.ndarray, y: np.ndarray, 
                                cv_folds: int = 5) -> Dict[str, float]:
        """Perform k-fold cross-validation."""
        classifier = self.model_configs[self.model_type]["classifier"]
        
        # Stratified k-fold to maintain class distribution
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Cross-validation scores
        cv_scores = cross_val_score(classifier, X, y, cv=skf, scoring='accuracy')
        
        return {
            "accuracy_mean": float(np.mean(cv_scores)),
            "accuracy_std": float(np.std(cv_scores)),
            "accuracy_scores": cv_scores.tolist()
        }

    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive classification metrics."""
        # Classification report
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "accuracy": float(report['accuracy']),
            "macro_precision": float(report['macro avg']['precision']),
            "macro_recall": float(report['macro avg']['recall']),
            "macro_f1": float(report['macro avg']['f1-score']),
            "weighted_precision": float(report['weighted avg']['precision']),
            "weighted_recall": float(report['weighted avg']['recall']),
            "weighted_f1": float(report['weighted avg']['f1-score'])
        }

    def predict(self, features: GestureFeatures) -> Tuple[str, float]:
        """Predict gesture class and confidence."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Normalize features
        normalized_features = self.feature_extractor.normalize_features(features)
        
        # Convert to feature vector
        feature_vector = self._features_to_vector(normalized_features)
        X = feature_vector.reshape(1, -1)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        y_pred = self.model.predict(X_scaled)[0]
        y_proba = self.model.predict_proba(X_scaled)[0]
        
        # Get class name and confidence
        class_name = self.label_encoder.inverse_transform([y_pred])[0]
        confidence = float(np.max(y_proba))
        
        return class_name, confidence

    def predict(self, features) -> Dict[str, Any]:
        """Predict gesture class and confidence. Works with both GestureFeatures and raw arrays."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Handle different input types
        if isinstance(features, list):
            # Raw feature array
            feature_vector = np.array(features)
        else:
            # GestureFeatures object
            normalized_features = self.feature_extractor.normalize_features(features)
            feature_vector = self._features_to_vector(normalized_features)
        
        X = feature_vector.reshape(1, -1)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Predict
        y_pred = self.model.predict(X_scaled)[0]
        y_proba = self.model.predict_proba(X_scaled)[0]
        
        # Get class name and confidence
        class_name = self.label_encoder.inverse_transform([y_pred])[0]
        confidence = float(np.max(y_proba))
        
        return {
            "predicted_class": class_name,
            "confidence": confidence,
            "probabilities": {
                self.label_encoder.inverse_transform([i])[0]: float(prob) 
                for i, prob in enumerate(y_proba)
            }
        }

    def predict_with_probabilities(self, features: GestureFeatures) -> Dict[str, float]:
        """Predict with probabilities for all classes."""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Normalize features
        normalized_features = self.feature_extractor.normalize_features(features)
        
        # Convert to feature vector
        feature_vector = self._features_to_vector(normalized_features)
        X = feature_vector.reshape(1, -1)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get probabilities
        y_proba = self.model.predict_proba(X_scaled)[0]
        class_names = self.label_encoder.classes_
        
        # Create probability dictionary
        probabilities = {}
        for class_name, prob in zip(class_names, y_proba):
            probabilities[class_name] = float(prob)
        
        return probabilities

    def save_model(self, filepath: str):
        """Save trained model to file."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "model_type": self.model_type,
            "is_trained": self.is_trained
        }
        
        joblib.dump(model_data, filepath)

    def load_model(self, filepath: str):
        """Load trained model from file."""
        model_data = joblib.load(filepath)
        
        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.label_encoder = model_data["label_encoder"]
        self.model_type = model_data["model_type"]
        self.is_trained = model_data["is_trained"]

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance for tree-based models."""
        if not self.is_trained or self.model_type not in ["random_forest"]:
            return None
        
        # Get feature importances
        classifier = self.model.named_steps['classifier']
        importances = classifier.feature_importances_
        
        # Create feature names
        feature_names = self._get_feature_names()
        
        # Create importance dictionary
        importance_dict = {}
        for name, importance in zip(feature_names, importances):
            importance_dict[name] = float(importance)
        
        # Sort by importance
        sorted_importance = dict(sorted(importance_dict.items(), 
                                      key=lambda x: x[1], reverse=True))
        
        return sorted_importance

    def _get_feature_names(self) -> List[str]:
        """Generate feature names for interpretability."""
        names = []
        
        # Finger angles
        fingers = ["thumb", "index", "middle", "ring", "pinky"]
        for finger in fingers:
            names.append(f"{finger}_angle")
        
        # Finger distances
        for finger in fingers:
            names.append(f"{finger}_distance")
        
        # Inter-finger angles
        for i in range(len(fingers) - 1):
            names.append(f"{fingers[i]}_{fingers[i+1]}_angle")
        
        # Fingertip distances (combinations)
        for i in range(len(fingers)):
            for j in range(i + 1, len(fingers)):
                names.append(f"{fingers[i]}_{fingers[j]}_distance")
        
        # Thumb-finger distances
        for finger in fingers[1:]:  # Exclude thumb itself
            names.append(f"thumb_{finger}_distance")
        
        # Hand properties
        names.extend(["hand_size", "palm_x", "palm_y", "palm_z", 
                     "palm_normal_x", "palm_normal_y", "palm_normal_z"])
        
        # Landmark positions
        for i in range(21):
            names.extend([f"landmark_{i}_x", f"landmark_{i}_y", f"landmark_{i}_z"])
        
        # Velocity features
        for i in range(21):
            names.extend([f"velocity_{i}_x", f"velocity_{i}_y", f"velocity_{i}_z"])
        
        # Acceleration features
        for i in range(21):
            names.extend([f"acceleration_{i}_x", f"acceleration_{i}_y", f"acceleration_{i}_z"])
        
        # Confidence scores
        names.extend(["detection_confidence", "tracking_confidence"])
        
        return names
