import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import time
import math

class ModernOverlay:
    """Modern, crisp overlay system with enhanced UX features."""
    
    def __init__(self):
        self.fps_history = []
        self.fps_update_time = time.time()
        self.gesture_history = []
        self.confidence_history = {}
        self.heatmap_data = {}
        self.animation_frame = 0
        
        # Fonts and UI styling
        self.font_large = cv2.FONT_HERSHEY_DUPLEX
        self.font_medium = cv2.FONT_HERSHEY_SIMPLEX
        self.font_small = cv2.FONT_HERSHEY_PLAIN
        
        # Enhanced color palette
        self.colors = {
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'green': (0, 255, 0),
            'red': (0, 0, 255),
            'blue': (255, 0, 0),
            'yellow': (0, 255, 255),
            'cyan': (255, 255, 0),
            'magenta': (255, 0, 255),
            'orange': (0, 165, 255),
            'purple': (128, 0, 255),
            'teal': (128, 128, 0),
            'lime': (0, 255, 128),
            'pink': (203, 192, 255),
            'dark_bg': (40, 40, 40),
            'light_bg': (220, 220, 220),
            'success': (34, 139, 34),
            'warning': (0, 140, 255),
            'danger': (0, 50, 200),
            'info': (255, 180, 0),
            'accent': (100, 200, 255)
        }
        
        # UI state
        self.show_advanced_hud = True
        self.show_gesture_trail = True
        self.show_confidence_bars = True
        self.show_heatmap = True

    def draw_rounded_rectangle(self, frame, pt1, pt2, color, thickness=-1, radius=10):
        """Draw a rounded rectangle."""
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Create mask for rounded corners
        mask = np.zeros((y2-y1, x2-x1), dtype=np.uint8)
        cv2.rectangle(mask, (radius, 0), (x2-x1-radius, y2-y1), 255, -1)
        cv2.rectangle(mask, (0, radius), (x2-x1, y2-y1-radius), 255, -1)
        cv2.circle(mask, (radius, radius), radius, 255, -1)
        cv2.circle(mask, (x2-x1-radius, radius), radius, 255, -1)
        cv2.circle(mask, (radius, y2-y1-radius), radius, 255, -1)
        cv2.circle(mask, (x2-x1-radius, y2-y1-radius), radius, 255, -1)
        
        # Apply rounded rectangle
        if thickness == -1:
            roi = frame[y1:y2, x1:x2]
            roi[mask == 255] = color
        else:
            cv2.rectangle(frame, pt1, pt2, color, thickness)

    def draw_progress_bar(self, frame, x, y, width, height, progress, color, bg_color=None):
        """Draw a modern progress bar."""
        if bg_color is None:
            bg_color = self.colors['dark_bg']
            
        # Background
        self.draw_rounded_rectangle(frame, (x, y), (x + width, y + height), bg_color, -1, 3)
        
        # Progress fill
        if progress > 0:
            fill_width = int(width * min(progress, 1.0))
            self.draw_rounded_rectangle(frame, (x, y), (x + fill_width, y + height), color, -1, 3)
            
        # Border
        self.draw_rounded_rectangle(frame, (x, y), (x + width, y + height), self.colors['white'], 1, 3)

    def draw_text_with_background(self, frame, text, pos, font, scale, color, bg_color=None, padding=5):
        """Draw text with a background panel."""
        x, y = pos
        (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, 2)
        
        if bg_color:
            self.draw_rounded_rectangle(
                frame, 
                (x - padding, y - text_height - padding),
                (x + text_width + padding, y + baseline + padding),
                bg_color, -1, 5
            )
        
        cv2.putText(frame, text, (x, y), font, scale, color, 2, cv2.LINE_AA)
        return text_width, text_height

    def update_fps(self, current_time):
        """Update FPS calculation with smoothing."""
        if len(self.fps_history) > 0:
            frame_time = current_time - self.fps_update_time
            if frame_time > 0:
                fps = 1.0 / frame_time
                self.fps_history.append(fps)
                if len(self.fps_history) > 10:
                    self.fps_history.pop(0)
        
        self.fps_update_time = current_time
        return sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0

    def add_gesture_to_history(self, gesture_name, confidence):
        """Add gesture to history for trail visualization."""
        current_time = time.time()
        self.gesture_history.append({
            'name': gesture_name,
            'confidence': confidence,
            'time': current_time
        })
        
        # Keep only recent gestures
        cutoff_time = current_time - 3.0  # 3 seconds
        self.gesture_history = [g for g in self.gesture_history if g['time'] > cutoff_time]

    def draw_main_hud(self, frame, mode_text: str, mode_color: tuple, fps: float, 
                     pinch_thr: float = None, smoothing: int = None, plugin_mode: bool = False):
        """Draw the main HUD with modern styling."""
        h, w = frame.shape[:2]
        current_time = time.time()
        self.animation_frame += 1
        
        # Update FPS
        avg_fps = self.update_fps(current_time)
        
        # Main status panel (top-left)
        panel_height = 120 if plugin_mode else 140
        self.draw_rounded_rectangle(frame, (10, 10), (350, panel_height), self.colors['dark_bg'], -1, 10)
        
        # Mode indicator with accent color
        mode_bg_color = self.colors['success'] if 'Plugin' in mode_text else self.colors['info']
        self.draw_text_with_background(frame, mode_text, (20, 35), self.font_large, 0.8, 
                                     self.colors['white'], mode_bg_color, 8)
        
        # FPS with color coding
        fps_color = self.colors['success'] if avg_fps > 25 else self.colors['warning'] if avg_fps > 15 else self.colors['danger']
        self.draw_text_with_background(frame, f"FPS: {avg_fps:5.1f}", (20, 65), self.font_medium, 0.6,
                                     fps_color, self.colors['dark_bg'])
        
        # Plugin mode specific info
        if plugin_mode:
            cv2.putText(frame, "Press 'p' to toggle Legacy Mode", (20, 90), self.font_small, 1.2, 
                       self.colors['accent'], 1, cv2.LINE_AA)
        else:
            # Legacy mode controls
            if pinch_thr is not None and smoothing is not None:
                cv2.putText(frame, f"Pinch Threshold: {pinch_thr:.3f}", (20, 90), self.font_small, 1.2,
                           self.colors['white'], 1, cv2.LINE_AA)
                cv2.putText(frame, f"Smoothing: {smoothing}", (20, 110), self.font_small, 1.2,
                           self.colors['white'], 1, cv2.LINE_AA)
                cv2.putText(frame, "[/] threshold  -/+ smoothing", (20, 130), self.font_small, 1.0,
                           self.colors['light_bg'], 1, cv2.LINE_AA)
        
        # Key shortcuts panel (bottom-left)
        shortcuts_y = h - 80
        self.draw_rounded_rectangle(frame, (10, shortcuts_y), (400, h - 10), self.colors['dark_bg'], -1, 8)
        
        shortcuts = [
            "q: Quit  |  m: Mode Toggle  |  c: Calibrate",
            "p: Plugin/Legacy Toggle  |  h: Hide/Show HUD"
        ]
        
        for i, shortcut in enumerate(shortcuts):
            cv2.putText(frame, shortcut, (20, shortcuts_y + 25 + i * 20), self.font_small, 1.1,
                       self.colors['accent'], 1, cv2.LINE_AA)

    def draw_gesture_status(self, frame, detected_gestures: List[Dict], active_actions: List[str] = None):
        """Draw current gesture status with confidence indicators."""
        h, w = frame.shape[:2]
        
        if not detected_gestures and not active_actions:
            return
            
        # Gesture status panel (top-right)
        panel_width = 280
        panel_x = w - panel_width - 10
        panel_height = max(100, 30 + len(detected_gestures) * 25 + (len(active_actions or []) * 20))
        
        self.draw_rounded_rectangle(frame, (panel_x, 10), (w - 10, panel_height + 10), 
                                  self.colors['dark_bg'], -1, 10)
        
        # Header
        self.draw_text_with_background(frame, "DETECTED GESTURES", (panel_x + 10, 35), 
                                     self.font_medium, 0.6, self.colors['accent'])
        
        y_offset = 60
        
        # Draw detected gestures with confidence bars
        for gesture in detected_gestures:
            gesture_name = gesture.get('name', 'Unknown')
            confidence = gesture.get('confidence', 0.0)
            hand = gesture.get('hand', 'Unknown')
            
            # Gesture name and hand
            text = f"{gesture_name} ({hand})"
            cv2.putText(frame, text, (panel_x + 15, y_offset), self.font_small, 1.2,
                       self.colors['white'], 1, cv2.LINE_AA)
            
            # Confidence bar
            bar_y = y_offset + 8
            confidence_color = self.colors['success'] if confidence > 0.8 else \
                              self.colors['warning'] if confidence > 0.5 else self.colors['danger']
            
            self.draw_progress_bar(frame, panel_x + 15, bar_y, 200, 8, confidence, confidence_color)
            
            # Confidence percentage
            cv2.putText(frame, f"{confidence:.1%}", (panel_x + 225, y_offset), self.font_small, 1.0,
                       confidence_color, 1, cv2.LINE_AA)
            
            y_offset += 25
            
        # Active actions
        if active_actions:
            y_offset += 10
            cv2.putText(frame, "ACTIVE ACTIONS:", (panel_x + 10, y_offset), self.font_small, 1.2,
                       self.colors['lime'], 1, cv2.LINE_AA)
            y_offset += 20
            
            for action in active_actions:
                cv2.putText(frame, f"• {action}", (panel_x + 15, y_offset), self.font_small, 1.1,
                           self.colors['lime'], 1, cv2.LINE_AA)
                y_offset += 18

    def draw_hand_landmarks_enhanced(self, frame, landmarks, hand_type="Unknown", connections=None):
        """Draw hand landmarks with enhanced visualization."""
        if not landmarks:
            return
            
        h, w = frame.shape[:2]
        
        # Convert normalized coordinates to pixel coordinates
        points = []
        for landmark in landmarks:
            x = int(landmark[0] * w)
            y = int(landmark[1] * h)
            z = landmark[2] if len(landmark) > 2 else 0
            points.append((x, y, z))
        
        # Draw connections if provided
        if connections:
            for connection in connections:
                start_idx, end_idx = connection
                if start_idx < len(points) and end_idx < len(points):
                    start_point = points[start_idx][:2]
                    end_point = points[end_idx][:2]
                    cv2.line(frame, start_point, end_point, self.colors['cyan'], 2)
        
        # Draw landmarks with different colors for different finger parts
        landmark_colors = {
            'thumb': self.colors['red'],
            'index': self.colors['green'], 
            'middle': self.colors['blue'],
            'ring': self.colors['yellow'],
            'pinky': self.colors['magenta'],
            'palm': self.colors['white']
        }
        
        # Finger tip indices
        finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
        finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
        
        for i, (x, y, z) in enumerate(points):
            # Determine color based on landmark
            if i in finger_tips:
                finger_idx = finger_tips.index(i)
                color = landmark_colors[finger_names[finger_idx]]
                radius = 6
            elif i == 0:  # Wrist
                color = landmark_colors['palm']
                radius = 8
            else:
                color = landmark_colors['palm']
                radius = 4
                
            # Draw landmark with glow effect
            cv2.circle(frame, (x, y), radius + 2, self.colors['black'], -1)
            cv2.circle(frame, (x, y), radius, color, -1)
            cv2.circle(frame, (x, y), radius - 1, self.colors['white'], 1)

    def draw_gesture_trainer_panel(self, frame, target_gesture: str, current_confidence: float, 
                                 training_progress: Dict):
        """Draw gesture training interface."""
        h, w = frame.shape[:2]
        
        # Training panel (center-bottom)
        panel_width = 500
        panel_height = 120
        panel_x = (w - panel_width) // 2
        panel_y = h - panel_height - 20
        
        # Semi-transparent background
        overlay = frame.copy()
        self.draw_rounded_rectangle(overlay, (panel_x, panel_y), 
                                  (panel_x + panel_width, panel_y + panel_height),
                                  self.colors['dark_bg'], -1, 15)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Header
        header_text = f"GESTURE TRAINER: {target_gesture.upper()}"
        self.draw_text_with_background(frame, header_text, (panel_x + 20, panel_y + 30),
                                     self.font_large, 0.7, self.colors['accent'])
        
        # Current confidence
        conf_text = f"Confidence: {current_confidence:.1%}"
        conf_color = self.colors['success'] if current_confidence > 0.8 else \
                    self.colors['warning'] if current_confidence > 0.5 else self.colors['danger']
        
        cv2.putText(frame, conf_text, (panel_x + 20, panel_y + 60), self.font_medium, 0.6,
                   conf_color, 2, cv2.LINE_AA)
        
        # Progress bar
        self.draw_progress_bar(frame, panel_x + 180, panel_y + 45, 280, 15, current_confidence, conf_color)
        
        # Training tips
        tip_text = "Hold the gesture steady for best results"
        cv2.putText(frame, tip_text, (panel_x + 20, panel_y + 90), self.font_small, 1.2,
                   self.colors['white'], 1, cv2.LINE_AA)

# Legacy function wrappers for backward compatibility
def draw_hud(frame, mode_text: str, mode_color: tuple[int, int, int], fps: float, 
             pinch_thr: float, smoothing: int):
    """Legacy HUD function - redirects to modern overlay."""
    overlay = ModernOverlay()
    overlay.draw_main_hud(frame, mode_text, mode_color, fps, pinch_thr, smoothing, plugin_mode=False)

def draw_status(frame, text: str, frame_height: int):
    """Legacy status function."""
    if not text:
        return
    cv2.putText(frame, text, (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

def draw_calibration_panel(frame, frame_width: int):
    """Enhanced calibration panel."""
    overlay = ModernOverlay()
    h, w = frame.shape[:2]
    
    # Modern calibration panel
    panel_width = min(w - 40, 600)
    panel_height = 160
    panel_x = (w - panel_width) // 2
    panel_y = (h - panel_height) // 2
    
    # Semi-transparent background
    bg_overlay = frame.copy()
    overlay.draw_rounded_rectangle(bg_overlay, (panel_x, panel_y), 
                                 (panel_x + panel_width, panel_y + panel_height),
                                 overlay.colors['dark_bg'], -1, 15)
    cv2.addWeighted(bg_overlay, 0.9, frame, 0.1, 0, frame)
    
    # Border
    overlay.draw_rounded_rectangle(frame, (panel_x, panel_y), 
                                 (panel_x + panel_width, panel_y + panel_height),
                                 overlay.colors['accent'], 3, 15)
    
    # Header
    overlay.draw_text_with_background(frame, "CALIBRATION MODE", (panel_x + 20, panel_y + 35),
                                    overlay.font_large, 0.8, overlay.colors['accent'])
    
    # Instructions
    instructions = [
        "1: Sample PINCH CLOSED (thumb-index touching)",
        "2: Sample PINCH OPEN (thumb-index apart)", 
        "S: Save & Exit  |  ESC: Exit without saving"
    ]
    
    for i, instruction in enumerate(instructions):
        y_pos = panel_y + 70 + i * 25
        cv2.putText(frame, instruction, (panel_x + 30, y_pos), overlay.font_medium, 0.6,
                   overlay.colors['white'], 1, cv2.LINE_AA)
