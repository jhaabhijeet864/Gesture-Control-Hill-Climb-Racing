#!/usr/bin/env python3
"""
Gesture Trainer - Interactive gesture training and calibration system
"""

import cv2
import numpy as np
import time
import json
import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from capture import VideoCapture
from inference import HandInference
from overlay import ModernOverlay

class TrainingState(Enum):
    IDLE = "idle"
    RECORDING = "recording"
    ANALYZING = "analyzing"
    COMPLETE = "complete"

@dataclass
class GestureDataPoint:
    """Single gesture sample data point."""
    landmarks: List[List[float]]
    timestamp: float
    confidence: float
    hand_type: str
    gesture_name: str

@dataclass
class TrainingSession:
    """Complete training session for a gesture."""
    gesture_name: str
    target_samples: int
    collected_samples: List[GestureDataPoint]
    start_time: float
    end_time: Optional[float] = None
    quality_score: float = 0.0
    
    @property
    def progress(self) -> float:
        return len(self.collected_samples) / self.target_samples
    
    @property
    def is_complete(self) -> bool:
        return len(self.collected_samples) >= self.target_samples

class GestureTrainer:
    """Interactive gesture training system with real-time feedback."""
    
    def __init__(self):
        self.overlay = ModernOverlay()
        self.capture = VideoCapture()
        self.inference = HandInference()
        
        self.current_session: Optional[TrainingSession] = None
        self.training_state = TrainingState.IDLE
        self.last_sample_time = 0
        self.sample_interval = 0.1  # Minimum time between samples
        
        # Training parameters
        self.min_confidence = 0.7
        self.stability_threshold = 0.05  # Max movement between samples
        self.quality_window = []  # Rolling window for quality assessment
        
        # UI state
        self.countdown_timer = 0
        self.feedback_message = ""
        self.feedback_color = (255, 255, 255)
        
        # Pre-defined gestures to train
        self.available_gestures = [
            "fist", "open_palm", "point", "peace", "thumbs_up",
            "pinch", "ok_sign", "rock", "scissors", "paper"
        ]
        
        self.gesture_descriptions = {
            "fist": "Close your hand into a fist",
            "open_palm": "Show open palm facing camera",
            "point": "Point with index finger extended",
            "peace": "Make peace sign with index and middle finger",
            "thumbs_up": "Show thumbs up gesture",
            "pinch": "Pinch thumb and index finger together",
            "ok_sign": "Make OK sign with thumb and index",
            "rock": "Make rock gesture (fist with thumb out)",
            "scissors": "Make scissors with index and middle finger",
            "paper": "Show flat open hand"
        }

    def start_training_session(self, gesture_name: str, target_samples: int = 50):
        """Start a new training session for a specific gesture."""
        if gesture_name not in self.available_gestures:
            raise ValueError(f"Unknown gesture: {gesture_name}")
        
        self.current_session = TrainingSession(
            gesture_name=gesture_name,
            target_samples=target_samples,
            collected_samples=[],
            start_time=time.time()
        )
        
        self.training_state = TrainingState.RECORDING
        self.countdown_timer = 3.0  # 3 second countdown
        self.feedback_message = f"Get ready to show: {gesture_name}"
        self.feedback_color = self.overlay.colors['warning']
        
        print(f"Starting training session for '{gesture_name}' - Need {target_samples} samples")

    def assess_sample_quality(self, landmarks: List[List[float]], hand_type: str) -> float:
        """Assess the quality of a gesture sample."""
        if not landmarks or len(landmarks) != 21:
            return 0.0
        
        # Check landmark stability (low variance = better quality)
        if len(self.quality_window) > 0:
            last_landmarks = self.quality_window[-1]
            movement = 0.0
            for i, (curr, prev) in enumerate(zip(landmarks, last_landmarks)):
                movement += sum((c - p) ** 2 for c, p in zip(curr[:2], prev[:2]))
            
            stability = max(0, 1.0 - (movement / len(landmarks)))
        else:
            stability = 1.0
        
        # Check hand centering (penalty for hands too close to edges)
        center_x = np.mean([lm[0] for lm in landmarks])
        center_y = np.mean([lm[1] for lm in landmarks])
        
        center_quality = 1.0
        if center_x < 0.2 or center_x > 0.8:
            center_quality *= 0.7
        if center_y < 0.2 or center_y > 0.8:
            center_quality *= 0.7
        
        # Check landmark spread (too compact = lower quality)
        x_spread = max([lm[0] for lm in landmarks]) - min([lm[0] for lm in landmarks])
        y_spread = max([lm[1] for lm in landmarks]) - min([lm[1] for lm in landmarks])
        spread_quality = min(1.0, (x_spread + y_spread) / 0.3)
        
        overall_quality = stability * center_quality * spread_quality
        return min(1.0, max(0.0, overall_quality))

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Process a single frame during training. Returns (annotated_frame, should_continue)."""
        current_time = time.time()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.inference.process(rgb)
        
        # Update countdown
        if self.countdown_timer > 0:
            self.countdown_timer = max(0, self.countdown_timer - 1/30)  # Assume 30 FPS
        
        hand_landmarks = None
        hand_type = "Unknown"
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            if results.multi_handedness:
                hand_type = results.multi_handedness[0].classification[0].label
        
        # Draw enhanced landmarks
        if hand_landmarks:
            landmarks_list = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
            self.overlay.draw_hand_landmarks_enhanced(frame, landmarks_list, hand_type)
            
            # Training logic
            if self.current_session and self.training_state == TrainingState.RECORDING:
                if self.countdown_timer <= 0:
                    # Assess sample quality
                    quality = self.assess_sample_quality(landmarks_list, hand_type)
                    
                    # Add to quality window for stability tracking
                    self.quality_window.append(landmarks_list)
                    if len(self.quality_window) > 10:
                        self.quality_window.pop(0)
                    
                    # Collect sample if quality is good and enough time has passed
                    if (quality >= self.min_confidence and 
                        current_time - self.last_sample_time >= self.sample_interval):
                        
                        sample = GestureDataPoint(
                            landmarks=landmarks_list,
                            timestamp=current_time,
                            confidence=quality,
                            hand_type=hand_type,
                            gesture_name=self.current_session.gesture_name
                        )
                        
                        self.current_session.collected_samples.append(sample)
                        self.last_sample_time = current_time
                        
                        # Update feedback
                        remaining = self.current_session.target_samples - len(self.current_session.collected_samples)
                        self.feedback_message = f"Sample {len(self.current_session.collected_samples)}/{self.current_session.target_samples} - Quality: {quality:.2f}"
                        self.feedback_color = self.overlay.colors['success']
                        
                        # Check if session complete
                        if self.current_session.is_complete:
                            self.training_state = TrainingState.COMPLETE
                            self.current_session.end_time = current_time
                            self.current_session.quality_score = np.mean([s.confidence for s in self.current_session.collected_samples])
                            self.feedback_message = f"Training complete! Average quality: {self.current_session.quality_score:.2f}"
                            self.feedback_color = self.overlay.colors['success']
                    
                    else:
                        # Provide guidance for better samples
                        if quality < self.min_confidence:
                            self.feedback_message = f"Hold steady - Quality: {quality:.2f} (need {self.min_confidence:.2f})"
                            self.feedback_color = self.overlay.colors['warning']
        
        # Draw training UI
        self.draw_training_ui(frame)
        
        return frame, self.training_state != TrainingState.COMPLETE

    def draw_training_ui(self, frame: np.ndarray):
        """Draw the training interface overlay."""
        h, w = frame.shape[:2]
        
        if not self.current_session:
            # Gesture selection menu
            self.draw_gesture_menu(frame)
            return
        
        # Training panel
        panel_width = 600
        panel_height = 200
        panel_x = (w - panel_width) // 2
        panel_y = 50
        
        # Semi-transparent background
        overlay = frame.copy()
        self.overlay.draw_rounded_rectangle(overlay, (panel_x, panel_y), 
                                          (panel_x + panel_width, panel_y + panel_height),
                                          self.overlay.colors['dark_bg'], -1, 15)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        
        # Header
        gesture_name = self.current_session.gesture_name.replace('_', ' ').title()
        header_text = f"Training: {gesture_name}"
        self.overlay.draw_text_with_background(frame, header_text, (panel_x + 20, panel_y + 35),
                                             self.overlay.font_large, 0.8, self.overlay.colors['accent'])
        
        # Description
        description = self.gesture_descriptions.get(self.current_session.gesture_name, "")
        cv2.putText(frame, description, (panel_x + 20, panel_y + 65), self.overlay.font_medium, 0.6,
                   self.overlay.colors['white'], 1, cv2.LINE_AA)
        
        # Progress bar
        progress = self.current_session.progress
        progress_y = panel_y + 85
        self.overlay.draw_progress_bar(frame, panel_x + 20, progress_y, panel_width - 40, 20, 
                                     progress, self.overlay.colors['success'])
        
        # Progress text
        progress_text = f"Samples: {len(self.current_session.collected_samples)}/{self.current_session.target_samples}"
        cv2.putText(frame, progress_text, (panel_x + 20, progress_y + 35), self.overlay.font_medium, 0.6,
                   self.overlay.colors['white'], 1, cv2.LINE_AA)
        
        # Countdown or feedback
        if self.countdown_timer > 0:
            countdown_text = f"Get ready... {int(self.countdown_timer) + 1}"
            text_size = cv2.getTextSize(countdown_text, self.overlay.font_large, 1.5, 3)[0]
            text_x = (w - text_size[0]) // 2
            text_y = (h + text_size[1]) // 2
            cv2.putText(frame, countdown_text, (text_x, text_y), self.overlay.font_large, 1.5,
                       self.overlay.colors['warning'], 3, cv2.LINE_AA)
        else:
            # Feedback message
            cv2.putText(frame, self.feedback_message, (panel_x + 20, panel_y + 160), 
                       self.overlay.font_medium, 0.6, self.feedback_color, 2, cv2.LINE_AA)
        
        # Instructions
        if self.training_state == TrainingState.COMPLETE:
            instructions = "Press 'S' to save, 'R' to restart, or 'Q' to quit"
        else:
            instructions = "Press 'ESC' to cancel training"
        
        cv2.putText(frame, instructions, (panel_x + 20, panel_y + 185), self.overlay.font_small, 1.2,
                   self.overlay.colors['light_bg'], 1, cv2.LINE_AA)

    def draw_gesture_menu(self, frame: np.ndarray):
        """Draw gesture selection menu."""
        h, w = frame.shape[:2]
        
        # Menu panel
        menu_width = 500
        menu_height = 400
        menu_x = (w - menu_width) // 2
        menu_y = (h - menu_height) // 2
        
        # Background
        overlay = frame.copy()
        self.overlay.draw_rounded_rectangle(overlay, (menu_x, menu_y), 
                                          (menu_x + menu_width, menu_y + menu_height),
                                          self.overlay.colors['dark_bg'], -1, 15)
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
        
        # Title
        title_text = "Gesture Trainer - Select Gesture"
        self.overlay.draw_text_with_background(frame, title_text, (menu_x + 20, menu_y + 35),
                                             self.overlay.font_large, 0.8, self.overlay.colors['accent'])
        
        # Gesture list
        y_offset = menu_y + 70
        for i, gesture in enumerate(self.available_gestures[:8]):  # Show first 8
            gesture_text = f"{i+1}. {gesture.replace('_', ' ').title()}"
            cv2.putText(frame, gesture_text, (menu_x + 30, y_offset), self.overlay.font_medium, 0.6,
                       self.overlay.colors['white'], 1, cv2.LINE_AA)
            y_offset += 30
        
        # Instructions
        instructions = [
            "Press number key (1-8) to select gesture",
            "Press 'Q' to return to main application"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (menu_x + 30, menu_y + menu_height - 60 + i * 25), 
                       self.overlay.font_small, 1.2, self.overlay.colors['light_bg'], 1, cv2.LINE_AA)

    def save_session(self, filepath: str = None):
        """Save the current training session to file."""
        if not self.current_session or not self.current_session.is_complete:
            return False
        
        if filepath is None:
            os.makedirs("training_data", exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filepath = f"training_data/{self.current_session.gesture_name}_{timestamp}.json"
        
        # Convert session to dictionary
        session_data = asdict(self.current_session)
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        print(f"Training session saved to: {filepath}")
        return True

    def run_interactive_trainer(self):
        """Run the interactive gesture trainer application."""
        print("Gesture Trainer Started")
        print("Commands:")
        print("  Numbers 1-8: Select gesture to train")
        print("  S: Save completed session")
        print("  R: Restart current session")
        print("  ESC: Cancel current session")
        print("  Q: Quit trainer")
        
        try:
            while True:
                ret, frame = self.capture.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                
                # Process frame
                annotated_frame, should_continue = self.process_frame(frame)
                
                if not should_continue:
                    # Session complete, wait for user input
                    pass
                
                cv2.imshow("Game Glide - Gesture Trainer", annotated_frame)
                
                # Handle key input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == 27:  # ESC
                    self.current_session = None
                    self.training_state = TrainingState.IDLE
                    self.feedback_message = ""
                elif key == ord('s') and self.current_session and self.current_session.is_complete:
                    self.save_session()
                    self.current_session = None
                    self.training_state = TrainingState.IDLE
                elif key == ord('r') and self.current_session:
                    # Restart current session
                    gesture_name = self.current_session.gesture_name
                    target_samples = self.current_session.target_samples
                    self.start_training_session(gesture_name, target_samples)
                elif key in [ord(str(i)) for i in range(1, 9)] and not self.current_session:
                    # Select gesture to train
                    gesture_idx = int(chr(key)) - 1
                    if gesture_idx < len(self.available_gestures):
                        self.start_training_session(self.available_gestures[gesture_idx])
        
        finally:
            self.capture.release()
            cv2.destroyAllWindows()

def main():
    """Run the gesture trainer application."""
    trainer = GestureTrainer()
    trainer.run_interactive_trainer()

if __name__ == "__main__":
    main()
