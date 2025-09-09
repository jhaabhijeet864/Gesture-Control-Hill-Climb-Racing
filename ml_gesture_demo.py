#!/usr/bin/env python3
"""
ML Gesture Recognition Demo
Demonstrates the complete robust gesture recognition system
"""

import cv2
import numpy as np
import time
import argparse
import json
from typing import Dict, Any, List
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from robust_gesture_system import RobustGestureRecognitionSystem, GestureResult
from temporal_gesture_models import TemporalGestureSequence
from ml_gesture_recognition import FeatureExtractor

class GestureRecognitionDemo:
    """Interactive demo for ML gesture recognition system."""
    
    def __init__(self):
        self.system = RobustGestureRecognitionSystem()
        self.demo_mode = "calibration"  # calibration, training, recognition
        self.training_data = []
        self.temporal_sequences = []
        self.current_gesture_label = None
        self.recording_sequence = False
        self.sequence_buffer = []
        
        # Demo UI elements
        self.ui_elements = {
            'instructions': "",
            'status': "",
            'metrics': {},
            'gesture_result': None
        }

    def run_demo(self, camera_id: int = 0):
        """Run the interactive demo."""
        print("🎯 ML Gesture Recognition System Demo")
        print("=====================================")
        
        # Initialize camera
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"❌ Error: Cannot open camera {camera_id}")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("\n📋 Demo Instructions:")
        print("  ESC - Exit demo")
        print("  SPACE - Start/complete current step")
        print("  C - Start calibration")
        print("  T - Training mode")
        print("  R - Recognition mode")
        print("  S - Save models")
        print("  L - Load models")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process frame based on current mode
                if self.demo_mode == "calibration":
                    frame = self._handle_calibration_mode(frame)
                elif self.demo_mode == "training":
                    frame = self._handle_training_mode(frame)
                elif self.demo_mode == "recognition":
                    frame = self._handle_recognition_mode(frame)
                
                # Draw UI
                frame = self._draw_ui(frame)
                
                # Show frame
                cv2.imshow('ML Gesture Recognition Demo', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord(' '):  # SPACE
                    self._handle_space_key()
                elif key == ord('c'):  # Start calibration
                    self._start_calibration()
                elif key == ord('t'):  # Training mode
                    self._start_training_mode()
                elif key == ord('r'):  # Recognition mode
                    self._start_recognition_mode()
                elif key == ord('s'):  # Save models
                    self._save_models()
                elif key == ord('l'):  # Load models
                    self._load_models()
                elif key >= ord('1') and key <= ord('9'):  # Gesture labels
                    self._set_gesture_label(key - ord('0'))
                
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.system.cleanup()

    def _handle_calibration_mode(self, frame: np.ndarray) -> np.ndarray:
        """Handle calibration mode."""
        if not hasattr(self, 'calibration_started'):
            self.ui_elements['instructions'] = "Press 'C' to start calibration or 'L' to load existing"
            self.ui_elements['status'] = "Ready for calibration"
            return frame
        
        # Process calibration frame
        result = self.system.process_calibration_frame(frame)
        
        if result.get('status') == 'complete':
            self.ui_elements['status'] = "✅ Calibration Complete!"
            self.ui_elements['instructions'] = "Press 'T' for training or 'R' for recognition"
            self.demo_mode = "ready"
        elif result.get('status') == 'collecting':
            progress = result.get('progress', 0)
            self.ui_elements['status'] = f"Calibrating... {progress*100:.1f}%"
            self.ui_elements['instructions'] = result.get('message', '')
        elif result.get('status') == 'no_hand_detected':
            self.ui_elements['status'] = "👋 Please show your hand"
            self.ui_elements['instructions'] = result.get('message', '')
        
        return result.get('frame', frame)

    def _handle_training_mode(self, frame: np.ndarray) -> np.ndarray:
        """Handle training data collection."""
        if not self.system.is_calibrated:
            self.ui_elements['status'] = "❌ Calibration required first"
            self.ui_elements['instructions'] = "Press 'C' to calibrate"
            return frame
        
        # Process frame to extract features
        result = self.system.process_frame(frame)
        
        if result:
            # Store training data if gesture label is set
            if self.current_gesture_label is not None:
                features_list = list(result.filtered_features.values())
                self.training_data.append((features_list, f"gesture_{self.current_gesture_label}"))
                
                # For temporal training, collect sequences
                if self.recording_sequence:
                    self.sequence_buffer.append({
                        'features': features_list,
                        'timestamp': result.timestamp
                    })
        
        # Update UI
        gesture_counts = {}
        for _, label in self.training_data:
            gesture_counts[label] = gesture_counts.get(label, 0) + 1
        
        self.ui_elements['status'] = f"Training samples: {len(self.training_data)}"
        self.ui_elements['instructions'] = f"Current gesture: {self.current_gesture_label or 'None'} | Press 1-9 for gesture, SPACE to train"
        self.ui_elements['metrics'] = gesture_counts
        
        return self._draw_hand_landmarks(frame, result)

    def _handle_recognition_mode(self, frame: np.ndarray) -> np.ndarray:
        """Handle real-time gesture recognition."""
        if not (self.system.is_ml_trained or self.system.is_temporal_trained):
            self.ui_elements['status'] = "❌ No trained models available"
            self.ui_elements['instructions'] = "Press 'T' to train models"
            return frame
        
        # Process frame
        result = self.system.process_frame(frame)
        
        if result:
            self.ui_elements['gesture_result'] = result
            
            # Update status
            if result.is_stable and result.confidence > 0.7:
                self.ui_elements['status'] = f"🎯 Detected: {result.gesture.upper()}"
            else:
                self.ui_elements['status'] = f"👀 Analyzing... ({result.gesture})"
            
            self.ui_elements['instructions'] = f"Confidence: {result.confidence:.2f} | Method: {result.method}"
            
            # Get performance metrics
            self.ui_elements['metrics'] = self.system.get_performance_metrics()
        else:
            self.ui_elements['status'] = "👋 Show your hand"
            self.ui_elements['instructions'] = "Position hand in camera view"
        
        return self._draw_hand_landmarks(frame, result)

    def _draw_ui(self, frame: np.ndarray) -> np.ndarray:
        """Draw UI elements on frame."""
        h, w = frame.shape[:2]
        
        # Create UI overlay
        overlay = frame.copy()
        
        # Background panel
        cv2.rectangle(overlay, (10, 10), (w-10, 180), (0, 0, 0), -1)
        cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)
        
        # Title
        cv2.putText(frame, "ML Gesture Recognition Demo", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        
        # Mode indicator
        mode_color = (0, 255, 0) if self.demo_mode == "recognition" else (255, 255, 0)
        cv2.putText(frame, f"Mode: {self.demo_mode.upper()}", 
                   (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
        
        # Status
        cv2.putText(frame, f"Status: {self.ui_elements['status']}", 
                   (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Instructions
        cv2.putText(frame, self.ui_elements['instructions'], 
                   (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # System status indicators
        y_offset = 160
        if self.system.is_calibrated:
            cv2.putText(frame, "✓ Calibrated", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if self.system.is_ml_trained:
            cv2.putText(frame, "✓ ML Trained", (150, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if self.system.is_temporal_trained:
            cv2.putText(frame, "✓ Temporal Trained", (280, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Gesture result display
        if self.ui_elements['gesture_result']:
            result = self.ui_elements['gesture_result']
            
            # Gesture name and confidence
            gesture_text = f"{result.gesture.upper()}"
            confidence_text = f"{result.confidence:.2f}"
            
            # Choose color based on confidence
            if result.confidence > 0.8:
                color = (0, 255, 0)  # Green
            elif result.confidence > 0.6:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 128, 255)  # Orange
            
            # Draw gesture result
            cv2.putText(frame, gesture_text, (w-300, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(frame, confidence_text, (w-300, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Stability indicator
            if result.is_stable:
                cv2.circle(frame, (w-50, 50), 15, (0, 255, 0), -1)
            else:
                cv2.circle(frame, (w-50, 50), 15, (0, 255, 255), 2)
        
        # Performance metrics
        if self.ui_elements['metrics']:
            metrics = self.ui_elements['metrics']
            y_start = h - 100
            
            if isinstance(metrics, dict) and 'processing_fps' in metrics:
                # Performance metrics
                cv2.putText(frame, f"FPS: {metrics.get('processing_fps', 0):.1f}", 
                           (20, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(frame, f"Avg Confidence: {metrics.get('average_confidence', 0):.2f}", 
                           (20, y_start + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                # Training metrics (gesture counts)
                y_pos = y_start
                for gesture, count in metrics.items():
                    cv2.putText(frame, f"{gesture}: {count}", 
                               (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    y_pos += 20
        
        return frame

    def _draw_hand_landmarks(self, frame: np.ndarray, result: GestureResult = None) -> np.ndarray:
        """Draw hand landmarks if available."""
        if not result:
            return frame
        
        # This is a simplified version - in practice, you'd need the actual landmarks
        # For demo purposes, we'll just draw a simple hand indicator
        h, w = frame.shape[:2]
        center = (w//2, h//2)
        
        # Draw a simple hand representation
        cv2.circle(frame, center, 10, (0, 255, 0), -1)  # Palm center
        
        # Draw confidence as a circle around the center
        radius = int(30 + result.confidence * 50)
        cv2.circle(frame, center, radius, (0, 255, 0), 2)
        
        return frame

    def _handle_space_key(self):
        """Handle space key press."""
        if self.demo_mode == "calibration" and hasattr(self, 'calibration_started'):
            # Space during calibration - handled by calibration flow
            pass
        elif self.demo_mode == "training":
            # Train models
            self._train_models()
        elif self.demo_mode == "recognition":
            # Reset system or take screenshot
            self.system.adaptive_filter.reset_tracking()

    def _start_calibration(self):
        """Start calibration process."""
        result = self.system.start_calibration("demo_user")
        self.calibration_started = True
        self.demo_mode = "calibration"
        self.ui_elements['status'] = "Starting calibration..."
        print("🎯 Started calibration process")

    def _start_training_mode(self):
        """Enter training mode."""
        if not self.system.is_calibrated:
            print("❌ Calibration required before training")
            return
        
        self.demo_mode = "training"
        self.training_data = []
        self.temporal_sequences = []
        self.ui_elements['status'] = "Training mode activated"
        print("📚 Entered training mode")

    def _start_recognition_mode(self):
        """Enter recognition mode."""
        if not (self.system.is_ml_trained or self.system.is_temporal_trained):
            print("❌ No trained models available")
            return
        
        self.demo_mode = "recognition"
        self.system.start_realtime_processing()
        self.ui_elements['status'] = "Recognition mode activated"
        print("🎯 Entered recognition mode")

    def _set_gesture_label(self, label: int):
        """Set current gesture label for training."""
        self.current_gesture_label = label
        print(f"📝 Set gesture label to: {label}")

    def _train_models(self):
        """Train both ML and temporal models."""
        if len(self.training_data) < 10:
            print("❌ Need at least 10 training samples")
            return
        
        print("🔄 Training ML classifier...")
        ml_results = self.system.train_ml_classifier(self.training_data)
        
        if ml_results.get('success', True):  # Assume success if no explicit failure
            print(f"✅ ML classifier trained with accuracy: {ml_results.get('accuracy', 'N/A')}")
        
        # Create temporal sequences for temporal model training
        if len(self.training_data) >= 50:  # Need more data for temporal
            print("🔄 Preparing temporal sequences...")
            sequences = self._create_temporal_sequences()
            
            if sequences:
                print("🔄 Training temporal model...")
                temporal_results = self.system.train_temporal_model(sequences)
                
                if temporal_results.get('success', True):
                    print(f"✅ Temporal model trained with accuracy: {temporal_results.get('final_val_accuracy', 'N/A')}")

    def _create_temporal_sequences(self) -> List[TemporalGestureSequence]:
        """Create temporal sequences from training data."""
        sequences = []
        
        # Group training data by gesture
        gesture_groups = {}
        for features, label in self.training_data:
            if label not in gesture_groups:
                gesture_groups[label] = []
            gesture_groups[label].append(features)
        
        # Create sequences (simplified - in practice you'd have actual temporal data)
        for gesture, feature_lists in gesture_groups.items():
            if len(feature_lists) >= 10:  # Minimum sequence length
                # Create a sequence from the feature lists
                sequence = TemporalGestureSequence(
                    features=feature_lists[:30],  # Use up to 30 frames
                    label=gesture,
                    timestamps=[i * 0.033 for i in range(len(feature_lists[:30]))],  # 30 FPS
                    sequence_length=len(feature_lists[:30])
                )
                sequences.append(sequence)
        
        return sequences

    def _save_models(self):
        """Save all trained models."""
        success = self.system.save_models("demo_models")
        if success:
            print("✅ Models saved successfully")
            self.ui_elements['status'] = "Models saved"
        else:
            print("❌ Failed to save models")
            self.ui_elements['status'] = "Save failed"

    def _load_models(self):
        """Load existing models."""
        results = self.system.load_models("demo_models")
        
        loaded_items = [item for item, success in results.items() if success]
        if loaded_items:
            print(f"✅ Loaded: {', '.join(loaded_items)}")
            self.ui_elements['status'] = f"Loaded: {', '.join(loaded_items)}"
        else:
            print("❌ No models found to load")
            self.ui_elements['status'] = "No models found"

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="ML Gesture Recognition Demo")
    parser.add_argument("--camera", type=int, default=0, help="Camera ID (default: 0)")
    parser.add_argument("--config", type=str, help="Configuration file path")
    
    args = parser.parse_args()
    
    # Create and run demo
    demo = GestureRecognitionDemo()
    
    # Load config if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
        demo.system.config.update(config)
        print(f"📁 Loaded config from: {args.config}")
    
    try:
        demo.run_demo(args.camera)
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"❌ Demo error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
