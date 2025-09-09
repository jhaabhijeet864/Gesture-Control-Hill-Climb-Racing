"""
Built-in Gesture Plugins

Collection of common gesture recognition plugins using rule-based and ML approaches.
"""
import math
import numpy as np
from typing import List, Dict, Any
import mediapipe as mp

from .gesture_base import GesturePlugin, GestureResult, FeatureExtractor, GestureClassifier


class RuleBasedGesturePlugin(GesturePlugin):
    """Rule-based gesture recognition for basic gestures."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.mp_hands = mp.solutions.hands
        
        # Configurable thresholds
        self.fist_threshold = config.get('fist_threshold', 0.8)
        self.pinch_threshold = config.get('pinch_threshold', 0.05)
        self.point_threshold = config.get('point_threshold', 0.7)
        
    def detect(self, hand_landmarks, frame: np.ndarray, hand_type: str) -> List[GestureResult]:
        """Detect rule-based gestures."""
        results = []
        
        if not hand_landmarks:
            return results
        
        # Fist detection
        if self._is_fist(hand_landmarks):
            confidence = self._fist_confidence(hand_landmarks)
            results.append(GestureResult(
                f"fist_{hand_type}",
                confidence,
                {"hand": hand_type, "method": "rule_based"}
            ))
        
        # Pinch detection
        pinch_result = self._detect_pinch(hand_landmarks, hand_type)
        if pinch_result:
            results.append(pinch_result)
        
        # Point detection
        if self._is_pointing(hand_landmarks):
            confidence = self._point_confidence(hand_landmarks)
            results.append(GestureResult(
                f"point_{hand_type}",
                confidence,
                {"hand": hand_type, "method": "rule_based"}
            ))
        
        return results
    
    def _is_fist(self, landmarks) -> bool:
        """Check if hand is making a fist."""
        # Check if fingertips are below PIP joints
        finger_tips = [
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            self.mp_hands.HandLandmark.RING_FINGER_TIP,
            self.mp_hands.HandLandmark.PINKY_TIP
        ]
        
        finger_pips = [
            self.mp_hands.HandLandmark.INDEX_FINGER_PIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
            self.mp_hands.HandLandmark.RING_FINGER_PIP,
            self.mp_hands.HandLandmark.PINKY_PIP
        ]
        
        closed_count = 0
        for tip, pip in zip(finger_tips, finger_pips):
            if landmarks.landmark[tip].y > landmarks.landmark[pip].y:
                closed_count += 1
        
        return closed_count >= 3  # At least 3 fingers closed
    
    def _fist_confidence(self, landmarks) -> float:
        """Calculate confidence for fist gesture."""
        finger_tips = [
            self.mp_hands.HandLandmark.INDEX_FINGER_TIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            self.mp_hands.HandLandmark.RING_FINGER_TIP,
            self.mp_hands.HandLandmark.PINKY_TIP
        ]
        
        finger_pips = [
            self.mp_hands.HandLandmark.INDEX_FINGER_PIP,
            self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
            self.mp_hands.HandLandmark.RING_FINGER_PIP,
            self.mp_hands.HandLandmark.PINKY_PIP
        ]
        
        total_bend = 0
        for tip, pip in zip(finger_tips, finger_pips):
            bend_ratio = max(0, landmarks.landmark[tip].y - landmarks.landmark[pip].y)
            total_bend += bend_ratio
        
        return min(1.0, total_bend * 5)  # Scale to 0-1
    
    def _detect_pinch(self, landmarks, hand_type: str) -> GestureResult:
        """Detect thumb-finger pinch gestures."""
        thumb_tip = landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        index_tip = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        
        # Thumb-index pinch
        thumb_index_dist = math.hypot(
            thumb_tip.x - index_tip.x,
            thumb_tip.y - index_tip.y
        )
        
        # Thumb-middle pinch
        thumb_middle_dist = math.hypot(
            thumb_tip.x - middle_tip.x,
            thumb_tip.y - middle_tip.y
        )
        
        if thumb_index_dist < self.pinch_threshold:
            confidence = 1.0 - (thumb_index_dist / self.pinch_threshold)
            return GestureResult(
                f"pinch_thumb_index_{hand_type}",
                confidence,
                {
                    "hand": hand_type,
                    "method": "rule_based",
                    "distance": thumb_index_dist
                }
            )
        elif thumb_middle_dist < self.pinch_threshold:
            confidence = 1.0 - (thumb_middle_dist / self.pinch_threshold)
            return GestureResult(
                f"pinch_thumb_middle_{hand_type}",
                confidence,
                {
                    "hand": hand_type,
                    "method": "rule_based",
                    "distance": thumb_middle_dist
                }
            )
        
        return None
    
    def _is_pointing(self, landmarks) -> bool:
        """Check if hand is pointing (index finger extended, others closed)."""
        # Index finger extended
        index_tip = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_pip = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
        index_extended = index_tip.y < index_pip.y
        
        if not index_extended:
            return False
        
        # Other fingers closed
        other_tips = [
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            self.mp_hands.HandLandmark.RING_FINGER_TIP,
            self.mp_hands.HandLandmark.PINKY_TIP
        ]
        
        other_pips = [
            self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
            self.mp_hands.HandLandmark.RING_FINGER_PIP,
            self.mp_hands.HandLandmark.PINKY_PIP
        ]
        
        closed_count = 0
        for tip, pip in zip(other_tips, other_pips):
            if landmarks.landmark[tip].y > landmarks.landmark[pip].y:
                closed_count += 1
        
        return closed_count >= 2  # At least 2 other fingers closed
    
    def _point_confidence(self, landmarks) -> float:
        """Calculate confidence for pointing gesture."""
        # Distance between index tip and other fingertips
        index_tip = landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        
        other_tips = [
            self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
            self.mp_hands.HandLandmark.RING_FINGER_TIP,
            self.mp_hands.HandLandmark.PINKY_TIP
        ]
        
        total_separation = 0
        for tip in other_tips:
            dist = math.hypot(
                index_tip.x - landmarks.landmark[tip].x,
                index_tip.y - landmarks.landmark[tip].y
            )
            total_separation += dist
        
        return min(1.0, total_separation * 3)  # Scale to 0-1
    
    def get_gesture_names(self) -> List[str]:
        """Return list of supported gestures."""
        return [
            "fist_left", "fist_right",
            "pinch_thumb_index_left", "pinch_thumb_index_right",
            "pinch_thumb_middle_left", "pinch_thumb_middle_right",
            "point_left", "point_right"
        ]
    
    def get_config_schema(self) -> Dict[str, Any]:
        """Return configuration schema."""
        return {
            "enabled": {"type": "boolean", "default": True},
            "fist_threshold": {"type": "float", "default": 0.8, "min": 0.1, "max": 1.0},
            "pinch_threshold": {"type": "float", "default": 0.05, "min": 0.01, "max": 0.2},
            "point_threshold": {"type": "float", "default": 0.7, "min": 0.1, "max": 1.0}
        }


class HandGeometryFeatures(FeatureExtractor):
    """Extract geometric features from hand landmarks."""
    
    def extract(self, hand_landmarks) -> np.ndarray:
        """Extract feature vector from hand landmarks."""
        if not hand_landmarks:
            return np.zeros(self.get_feature_count())
        
        features = []
        landmarks = hand_landmarks.landmark
        
        # Finger lengths (tip to MCP)
        finger_indices = [
            (mp.solutions.hands.HandLandmark.THUMB_TIP, mp.solutions.hands.HandLandmark.THUMB_MCP),
            (mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP, mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP),
            (mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP, mp.solutions.hands.HandLandmark.MIDDLE_FINGER_MCP),
            (mp.solutions.hands.HandLandmark.RING_FINGER_TIP, mp.solutions.hands.HandLandmark.RING_FINGER_MCP),
            (mp.solutions.hands.HandLandmark.PINKY_TIP, mp.solutions.hands.HandLandmark.PINKY_MCP)
        ]
        
        for tip, mcp in finger_indices:
            length = math.hypot(
                landmarks[tip].x - landmarks[mcp].x,
                landmarks[tip].y - landmarks[mcp].y
            )
            features.append(length)
        
        # Finger angles (relative to palm)
        wrist = landmarks[mp.solutions.hands.HandLandmark.WRIST]
        for tip, mcp in finger_indices:
            # Vector from wrist to MCP
            palm_vec = (landmarks[mcp].x - wrist.x, landmarks[mcp].y - wrist.y)
            # Vector from MCP to tip
            finger_vec = (landmarks[tip].x - landmarks[mcp].x, landmarks[tip].y - landmarks[mcp].y)
            
            # Angle between vectors
            dot_product = palm_vec[0] * finger_vec[0] + palm_vec[1] * finger_vec[1]
            palm_mag = math.hypot(*palm_vec)
            finger_mag = math.hypot(*finger_vec)
            
            if palm_mag > 0 and finger_mag > 0:
                cos_angle = dot_product / (palm_mag * finger_mag)
                angle = math.acos(max(-1, min(1, cos_angle)))
                features.append(angle)
            else:
                features.append(0)
        
        # Inter-finger distances
        tip_landmarks = [
            mp.solutions.hands.HandLandmark.THUMB_TIP,
            mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP,
            mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP,
            mp.solutions.hands.HandLandmark.RING_FINGER_TIP,
            mp.solutions.hands.HandLandmark.PINKY_TIP
        ]
        
        for i in range(len(tip_landmarks)):
            for j in range(i + 1, len(tip_landmarks)):
                dist = math.hypot(
                    landmarks[tip_landmarks[i]].x - landmarks[tip_landmarks[j]].x,
                    landmarks[tip_landmarks[i]].y - landmarks[tip_landmarks[j]].y
                )
                features.append(dist)
        
        return np.array(features)
    
    def get_feature_names(self) -> List[str]:
        """Return feature names."""
        names = []
        
        # Finger lengths
        fingers = ["thumb", "index", "middle", "ring", "pinky"]
        for finger in fingers:
            names.append(f"{finger}_length")
        
        # Finger angles
        for finger in fingers:
            names.append(f"{finger}_angle")
        
        # Inter-finger distances
        for i in range(len(fingers)):
            for j in range(i + 1, len(fingers)):
                names.append(f"dist_{fingers[i]}_{fingers[j]}")
        
        return names
    
    def get_feature_count(self) -> int:
        """Return number of features."""
        return len(self.get_feature_names())
