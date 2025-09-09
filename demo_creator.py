#!/usr/bin/env python3
"""
Demo Creator - Generate screenshots, videos, and animations for Game Glide
"""

import cv2
import numpy as np
import os
import time
import json
from typing import List, Tuple, Optional
import subprocess
from datetime import datetime

from capture import VideoCapture
from inference import HandInference
from overlay import ModernOverlay
from plugins.plugin_manager import PluginManager

class DemoCreator:
    """Creates demonstration materials for Game Glide."""
    
    def __init__(self):
        self.overlay = ModernOverlay()
        self.capture = VideoCapture()
        self.inference = HandInference()
        self.plugin_manager = PluginManager("config.yaml")
        
        # Demo settings
        self.output_dir = "demos"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Video settings
        self.demo_fps = 30
        self.demo_duration = 10  # seconds
        
        # Create subdirectories
        for subdir in ["screenshots", "videos", "animations", "gifs"]:
            os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)

    def create_feature_showcase_screenshots(self):
        """Create screenshots showcasing different features."""
        print("Creating feature showcase screenshots...")
        
        scenarios = [
            {
                "name": "main_interface",
                "title": "Main Interface - Plugin Mode",
                "description": "Modern HUD with gesture detection",
                "setup": self.setup_main_interface
            },
            {
                "name": "gesture_detection",
                "title": "Gesture Detection",
                "description": "Real-time hand landmark detection",
                "setup": self.setup_gesture_detection
            },
            {
                "name": "profile_system",
                "title": "Profile System",
                "description": "Game-specific configuration profiles",
                "setup": self.setup_profile_system
            },
            {
                "name": "calibration_mode",
                "title": "Calibration Mode",
                "description": "Interactive gesture calibration",
                "setup": self.setup_calibration_mode
            }
        ]
        
        for scenario in scenarios:
            try:
                print(f"  Creating screenshot: {scenario['name']}")
                frame = self.create_scenario_frame(scenario)
                
                if frame is not None:
                    # Add title overlay
                    self.add_title_overlay(frame, scenario['title'], scenario['description'])
                    
                    # Save screenshot
                    filename = os.path.join(self.output_dir, "screenshots", f"{scenario['name']}.png")
                    cv2.imwrite(filename, frame)
                    print(f"    Saved: {filename}")
                
            except Exception as e:
                print(f"    Error creating {scenario['name']}: {e}")

    def create_scenario_frame(self, scenario) -> Optional[np.ndarray]:
        """Create a frame for a specific scenario."""
        # Capture base frame
        ret, frame = self.capture.read()
        if not ret:
            return None
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.inference.process(rgb)
        
        # Apply scenario-specific setup
        if scenario.get("setup"):
            frame = scenario["setup"](frame, results)
        
        return frame

    def setup_main_interface(self, frame, results):
        """Set up main interface demo."""
        # Draw modern HUD
        self.overlay.draw_main_hud(frame, "PLUGIN MODE", (0, 255, 255), 30.0, 
                                 plugin_mode=True)
        
        # Add some mock gesture detections
        mock_gestures = [
            {"name": "fist_right", "confidence": 0.85, "hand": "Right"},
            {"name": "point_left", "confidence": 0.72, "hand": "Left"}
        ]
        
        mock_actions = ["key: right (hold)", "mouse: move"]
        
        self.overlay.draw_gesture_status(frame, mock_gestures, mock_actions)
        
        # Draw hand landmarks if available
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                landmarks_list = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                hand_type = "Right" if idx == 0 else "Left"
                self.overlay.draw_hand_landmarks_enhanced(frame, landmarks_list, hand_type)
        
        return frame

    def setup_gesture_detection(self, frame, results):
        """Set up gesture detection demo."""
        # Focus on hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks_list = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                self.overlay.draw_hand_landmarks_enhanced(frame, landmarks_list, "Demo")
        
        # Add gesture info panel
        h, w = frame.shape[:2]
        panel_x = w - 300
        panel_y = 50
        
        self.overlay.draw_rounded_rectangle(frame, (panel_x, panel_y), (w - 20, panel_y + 200),
                                          self.overlay.colors['dark_bg'], -1, 10)
        
        # Title
        cv2.putText(frame, "GESTURE DETECTION", (panel_x + 15, panel_y + 30),
                   self.overlay.font_medium, 0.7, self.overlay.colors['accent'], 2)
        
        # Info
        info_lines = [
            "• 21 hand landmarks tracked",
            "• Real-time processing",
            "• Multi-hand support",
            "• Confidence scoring",
            "• Gesture classification"
        ]
        
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (panel_x + 15, panel_y + 65 + i * 25),
                       self.overlay.font_small, 1.1, self.overlay.colors['white'], 1)
        
        return frame

    def setup_profile_system(self, frame, results):
        """Set up profile system demo."""
        # Show profile info
        profile_info = {
            "name": "Hill Climb Racing",
            "gestures": 4,
            "mappings": 6,
            "confidence": "85%"
        }
        
        h, w = frame.shape[:2]
        panel_x = 50
        panel_y = 50
        panel_width = 350
        panel_height = 250
        
        self.overlay.draw_rounded_rectangle(frame, (panel_x, panel_y), 
                                          (panel_x + panel_width, panel_y + panel_height),
                                          self.overlay.colors['dark_bg'], -1, 10)
        
        # Title
        cv2.putText(frame, "ACTIVE PROFILE", (panel_x + 20, panel_y + 35),
                   self.overlay.font_large, 0.8, self.overlay.colors['accent'], 2)
        
        # Profile details
        y_offset = panel_y + 70
        for key, value in profile_info.items():
            cv2.putText(frame, f"{key.title()}: {value}", (panel_x + 20, y_offset),
                       self.overlay.font_medium, 0.6, self.overlay.colors['white'], 1)
            y_offset += 30
        
        # Mappings preview
        mappings = [
            "fist_right → gas pedal",
            "fist_left → brake",
            "point_right → boost",
            "open_palm → pause"
        ]
        
        cv2.putText(frame, "Gesture Mappings:", (panel_x + 20, y_offset + 10),
                   self.overlay.font_medium, 0.6, self.overlay.colors['lime'], 1)
        y_offset += 35
        
        for mapping in mappings:
            cv2.putText(frame, f"• {mapping}", (panel_x + 25, y_offset),
                       self.overlay.font_small, 1.0, self.overlay.colors['white'], 1)
            y_offset += 22
        
        return frame

    def setup_calibration_mode(self, frame, results):
        """Set up calibration mode demo."""
        # Use the enhanced calibration panel
        from overlay import draw_calibration_panel
        draw_calibration_panel(frame, frame.shape[1])
        
        return frame

    def add_title_overlay(self, frame, title, description):
        """Add title overlay to frame."""
        h, w = frame.shape[:2]
        
        # Title background
        title_height = 80
        overlay_bg = frame.copy()
        self.overlay.draw_rounded_rectangle(overlay_bg, (0, 0), (w, title_height),
                                          self.overlay.colors['dark_bg'], -1, 0)
        cv2.addWeighted(overlay_bg, 0.8, frame, 0.2, 0, frame)
        
        # Title text
        cv2.putText(frame, title, (20, 35), self.overlay.font_large, 1.0,
                   self.overlay.colors['accent'], 2, cv2.LINE_AA)
        
        # Description
        cv2.putText(frame, description, (20, 60), self.overlay.font_medium, 0.6,
                   self.overlay.colors['white'], 1, cv2.LINE_AA)
        
        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (w - 200, 25), self.overlay.font_small, 1.0,
                   self.overlay.colors['light_bg'], 1, cv2.LINE_AA)

    def create_gesture_demo_video(self, gesture_name: str, duration: int = 10):
        """Create a video demonstrating a specific gesture."""
        print(f"Creating demo video for gesture: {gesture_name}")
        
        # Video writer setup
        filename = os.path.join(self.output_dir, "videos", f"{gesture_name}_demo.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # Get frame dimensions
        ret, test_frame = self.capture.read()
        if not ret:
            print("Failed to get test frame")
            return
        
        h, w = test_frame.shape[:2]
        writer = cv2.VideoWriter(filename, fourcc, self.demo_fps, (w, h))
        
        frames_to_record = duration * self.demo_fps
        frame_count = 0
        
        print(f"  Recording {frames_to_record} frames...")
        
        try:
            while frame_count < frames_to_record:
                ret, frame = self.capture.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.inference.process(rgb)
                
                # Add demo overlay
                self.add_demo_video_overlay(frame, gesture_name, frame_count, frames_to_record)
                
                # Process gestures if hands detected
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        landmarks_list = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                        self.overlay.draw_hand_landmarks_enhanced(frame, landmarks_list, "Demo")
                        
                        # Show gesture trainer interface
                        confidence = np.random.uniform(0.6, 0.9)  # Mock confidence
                        self.overlay.draw_gesture_trainer_panel(frame, gesture_name, confidence, {})
                
                writer.write(frame)
                frame_count += 1
                
                if frame_count % (self.demo_fps * 2) == 0:
                    print(f"    Recorded {frame_count}/{frames_to_record} frames")
        
        finally:
            writer.release()
            print(f"  Video saved: {filename}")

    def add_demo_video_overlay(self, frame, gesture_name, frame_count, total_frames):
        """Add overlay for demo video."""
        h, w = frame.shape[:2]
        
        # Progress bar
        progress = frame_count / total_frames
        bar_width = 300
        bar_x = (w - bar_width) // 2
        bar_y = h - 50
        
        self.overlay.draw_progress_bar(frame, bar_x, bar_y, bar_width, 10, 
                                     progress, self.overlay.colors['accent'])
        
        # Demo info
        demo_text = f"Demo: {gesture_name.replace('_', ' ').title()}"
        cv2.putText(frame, demo_text, (20, h - 30), self.overlay.font_medium, 0.7,
                   self.overlay.colors['white'], 2, cv2.LINE_AA)
        
        # Frame counter
        frame_text = f"Frame {frame_count}/{total_frames}"
        cv2.putText(frame, frame_text, (w - 200, h - 30), self.overlay.font_small, 1.0,
                   self.overlay.colors['light_bg'], 1, cv2.LINE_AA)

    def create_feature_comparison_animation(self):
        """Create an animation comparing legacy vs plugin mode."""
        print("Creating feature comparison animation...")
        
        filename = os.path.join(self.output_dir, "animations", "feature_comparison.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        ret, test_frame = self.capture.read()
        if not ret:
            return
        
        h, w = test_frame.shape[:2]
        writer = cv2.VideoWriter(filename, fourcc, self.demo_fps, (w, h))
        
        # Animation phases
        phases = [
            {"name": "Legacy Mode", "duration": 3, "mode": "legacy"},
            {"name": "Transition", "duration": 1, "mode": "transition"},
            {"name": "Plugin Mode", "duration": 3, "mode": "plugin"},
            {"name": "Feature Highlight", "duration": 3, "mode": "highlight"}
        ]
        
        total_frames = sum(phase["duration"] * self.demo_fps for phase in phases)
        frame_count = 0
        
        try:
            for phase in phases:
                phase_frames = phase["duration"] * self.demo_fps
                for i in range(phase_frames):
                    ret, frame = self.capture.read()
                    if not ret:
                        break
                    
                    frame = cv2.flip(frame, 1)
                    
                    # Apply phase-specific overlay
                    self.add_comparison_overlay(frame, phase, i, phase_frames)
                    
                    writer.write(frame)
                    frame_count += 1
                    
                print(f"  Completed phase: {phase['name']}")
        
        finally:
            writer.release()
            print(f"  Animation saved: {filename}")

    def add_comparison_overlay(self, frame, phase, phase_frame, phase_total):
        """Add overlay for comparison animation."""
        h, w = frame.shape[:2]
        
        # Phase title
        title_text = f"Game Glide: {phase['name']}"
        cv2.putText(frame, title_text, (20, 40), self.overlay.font_large, 1.0,
                   self.overlay.colors['accent'], 2, cv2.LINE_AA)
        
        # Mode-specific UI
        if phase["mode"] == "legacy":
            # Simple legacy UI
            cv2.putText(frame, "GAME MODE", (20, 80), self.overlay.font_medium, 0.7,
                       (0, 0, 255), 2)
            cv2.putText(frame, "FPS: 30.0", (20, 110), self.overlay.font_medium, 0.6,
                       (255, 255, 0), 2)
            cv2.putText(frame, "Basic gesture control", (20, 140), self.overlay.font_small, 1.2,
                       (255, 255, 255), 1)
                       
        elif phase["mode"] == "plugin":
            # Modern plugin UI
            self.overlay.draw_main_hud(frame, "PLUGIN MODE", (0, 255, 255), 30.0, 
                                     plugin_mode=True)
            mock_gestures = [{"name": "fist_right", "confidence": 0.85, "hand": "Right"}]
            mock_actions = ["key: right (hold)"]
            self.overlay.draw_gesture_status(frame, mock_gestures, mock_actions)
            
        elif phase["mode"] == "transition":
            # Transition effect
            alpha = phase_frame / phase_total
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), self.overlay.colors['accent'], -1)
            cv2.addWeighted(overlay, alpha * 0.3, frame, 1 - alpha * 0.3, 0, frame)
            
            cv2.putText(frame, "UPGRADING...", (w//2 - 100, h//2), 
                       self.overlay.font_large, 1.2, self.overlay.colors['white'], 3)
                       
        elif phase["mode"] == "highlight":
            # Highlight features
            features = [
                "✓ Modern UI with confidence bars",
                "✓ Profile-based configurations", 
                "✓ Hot-reload capability",
                "✓ Enhanced gesture detection",
                "✓ Multi-backend support"
            ]
            
            panel_y = h // 2 - 100
            for i, feature in enumerate(features):
                cv2.putText(frame, feature, (50, panel_y + i * 30), 
                           self.overlay.font_medium, 0.6, self.overlay.colors['lime'], 2)

    def create_quick_demo_gif(self):
        """Create a quick animated GIF for social media."""
        print("Creating quick demo GIF...")
        
        # Record frames for GIF
        frames = []
        num_frames = 60  # 2 seconds at 30fps
        
        for i in range(num_frames):
            ret, frame = self.capture.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.inference.process(rgb)
            
            # Add quick demo overlay
            cv2.putText(frame, "Game Glide", (20, 50), self.overlay.font_large, 1.5,
                       self.overlay.colors['accent'], 3, cv2.LINE_AA)
            cv2.putText(frame, "Hand Gesture Control", (20, 90), self.overlay.font_medium, 0.8,
                       self.overlay.colors['white'], 2, cv2.LINE_AA)
            
            # Draw landmarks if available
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    landmarks_list = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                    self.overlay.draw_hand_landmarks_enhanced(frame, landmarks_list, "Demo")
            
            frames.append(frame)
        
        # Save as video first, then convert to GIF
        temp_video = os.path.join(self.output_dir, "temp_gif.mp4")
        gif_path = os.path.join(self.output_dir, "gifs", "quick_demo.gif")
        
        if frames:
            h, w = frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(temp_video, fourcc, 30, (w, h))
            
            for frame in frames:
                writer.write(frame)
            writer.release()
            
            # Convert to GIF using ffmpeg if available
            try:
                subprocess.run([
                    'ffmpeg', '-i', temp_video, '-vf', 
                    'fps=15,scale=480:-1:flags=lanczos', '-y', gif_path
                ], check=True, capture_output=True)
                os.remove(temp_video)
                print(f"  GIF saved: {gif_path}")
            except (subprocess.CalledProcessError, FileNotFoundError):
                print("  ffmpeg not available, keeping video file")
                os.rename(temp_video, gif_path.replace('.gif', '.mp4'))

    def create_all_demos(self):
        """Create all demonstration materials."""
        print("=== Game Glide Demo Creator ===")
        print("Creating comprehensive demonstration materials...")
        
        try:
            # Screenshots
            self.create_feature_showcase_screenshots()
            
            # Videos
            gestures_to_demo = ["fist", "point", "pinch", "open_palm"]
            for gesture in gestures_to_demo:
                self.create_gesture_demo_video(gesture, 8)
            
            # Animations
            self.create_feature_comparison_animation()
            
            # GIF
            self.create_quick_demo_gif()
            
            print("\n=== Demo Creation Complete ===")
            print(f"All files saved to: {os.path.abspath(self.output_dir)}")
            
            # Generate index file
            self.generate_demo_index()
            
        except Exception as e:
            print(f"Error during demo creation: {e}")
        
        finally:
            if hasattr(self, 'capture'):
                self.capture.release()

    def generate_demo_index(self):
        """Generate an index file listing all created demos."""
        index_path = os.path.join(self.output_dir, "demo_index.md")
        
        with open(index_path, 'w') as f:
            f.write("# Game Glide - Demonstration Materials\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Screenshots
            f.write("## Screenshots\n\n")
            screenshot_dir = os.path.join(self.output_dir, "screenshots")
            if os.path.exists(screenshot_dir):
                for file in os.listdir(screenshot_dir):
                    if file.endswith('.png'):
                        f.write(f"- ![{file}](screenshots/{file})\n")
            
            # Videos
            f.write("\n## Videos\n\n")
            video_dir = os.path.join(self.output_dir, "videos")
            if os.path.exists(video_dir):
                for file in os.listdir(video_dir):
                    if file.endswith('.mp4'):
                        f.write(f"- [{file}](videos/{file})\n")
            
            # Animations
            f.write("\n## Animations\n\n")
            animation_dir = os.path.join(self.output_dir, "animations")
            if os.path.exists(animation_dir):
                for file in os.listdir(animation_dir):
                    if file.endswith('.mp4'):
                        f.write(f"- [{file}](animations/{file})\n")
            
            # GIFs
            f.write("\n## GIFs\n\n")
            gif_dir = os.path.join(self.output_dir, "gifs")
            if os.path.exists(gif_dir):
                for file in os.listdir(gif_dir):
                    if file.endswith(('.gif', '.mp4')):
                        f.write(f"- [{file}](gifs/{file})\n")
        
        print(f"Demo index created: {index_path}")

def main():
    """Create demonstration materials."""
    creator = DemoCreator()
    creator.create_all_demos()

if __name__ == "__main__":
    main()
