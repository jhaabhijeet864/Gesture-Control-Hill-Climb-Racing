"""
Gesture Plugin Base Classes

Defines the interface for gesture recognition plugins that can be loaded dynamically.
Each plugin can define custom gestures, features, thresholds, and ML models.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import numpy as np


class GestureResult:
    """Result of gesture detection with confidence and metadata."""
    
    def __init__(self, gesture_name: str, confidence: float, metadata: Dict[str, Any] = None):
        self.gesture_name = gesture_name
        self.confidence = confidence
        self.metadata = metadata or {}
        self.timestamp = None


class GesturePlugin(ABC):
    """Base class for gesture recognition plugins."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = self.__class__.__name__
        self.enabled = config.get('enabled', True)
        
    @abstractmethod
    def detect(self, hand_landmarks, frame: np.ndarray, hand_type: str) -> List[GestureResult]:
        """
        Detect gestures from hand landmarks.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            frame: Current video frame
            hand_type: 'left' or 'right'
            
        Returns:
            List of detected gestures with confidence scores
        """
        pass
    
    @abstractmethod
    def get_gesture_names(self) -> List[str]:
        """Return list of gesture names this plugin can detect."""
        pass
    
    def configure(self, config: Dict[str, Any]):
        """Update plugin configuration."""
        self.config.update(config)
        self.enabled = self.config.get('enabled', True)
    
    def get_config_schema(self) -> Dict[str, Any]:
        """Return configuration schema for UI generation."""
        return {
            "enabled": {"type": "boolean", "default": True, "description": "Enable/disable plugin"}
        }


class FeatureExtractor(ABC):
    """Base class for feature extraction from hand landmarks."""
    
    @abstractmethod
    def extract(self, hand_landmarks) -> np.ndarray:
        """Extract feature vector from hand landmarks."""
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Return names of extracted features."""
        pass


class GestureClassifier(ABC):
    """Base class for gesture classification models."""
    
    @abstractmethod
    def predict(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Predict gesture from features.
        
        Returns:
            Tuple of (gesture_name, confidence)
        """
        pass
    
    @abstractmethod
    def train(self, features: np.ndarray, labels: List[str]):
        """Train the classifier with feature vectors and labels."""
        pass
    
    def save_model(self, path: str):
        """Save trained model to file."""
        pass
    
    def load_model(self, path: str):
        """Load trained model from file."""
        pass
