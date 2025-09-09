import math
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands

def is_fist(landmarks):
    index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    little_tip = landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

    if (index_tip.y > landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y and
        middle_tip.y > landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y and
        ring_tip.y > landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y and
        little_tip.y > landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y):
        return True
    return False

class ClickDetector:
    def __init__(self, pinch_threshold=0.05, ready_threshold=0.1, cooldown_frames=5):
        self.pinch_threshold = float(pinch_threshold)
        self.ready_threshold = float(ready_threshold)
        self.cooldown_frames = int(cooldown_frames)
        self.is_left_clicking = False
        self.is_right_clicking = False
        self.click_cooldown = 0

    def set_thresholds(self, pinch_threshold=None, ready_threshold=None, cooldown_frames=None):
        if pinch_threshold is not None:
            self.pinch_threshold = float(pinch_threshold)
        if ready_threshold is not None:
            self.ready_threshold = float(ready_threshold)
        if cooldown_frames is not None:
            self.cooldown_frames = int(cooldown_frames)

    def detect_click(self, landmarks, frame, mouse, Button, game_mode):
        # Returns (status_text, point or None)
        if game_mode:
            return "", None

        if self.click_cooldown > 0:
            self.click_cooldown -= 1
            return "", None

        thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

        thumb_index_distance = math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)
        thumb_middle_distance = math.hypot(thumb_tip.x - middle_tip.x, thumb_tip.y - middle_tip.y)

        thumb_x, thumb_y = int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0])
        index_x, index_y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])
        middle_x, middle_y = int(middle_tip.x * frame.shape[1]), int(middle_tip.y * frame.shape[0])

        # Left click - thumb and index pinch
        if thumb_index_distance < self.pinch_threshold:
            pinch_point = ((thumb_x + index_x) // 2, (thumb_y + index_y) // 2)
            cv2.circle(frame, pinch_point, 10, (0, 255, 0), -1)
            if not self.is_left_clicking and self.click_cooldown == 0:
                mouse.press(Button.left)
                self.is_left_clicking = True
                self.click_cooldown = self.cooldown_frames
                return "Left Click", pinch_point
        elif self.is_left_clicking:
            mouse.release(Button.left)
            self.is_left_clicking = False
            self.click_cooldown = self.cooldown_frames

        # Right click - thumb and middle pinch
        if thumb_middle_distance < self.pinch_threshold:
            pinch_point = ((thumb_x + middle_x) // 2, (thumb_y + middle_y) // 2)
            cv2.circle(frame, pinch_point, 10, (255, 0, 0), -1)
            if not self.is_right_clicking and self.click_cooldown == 0:
                mouse.press(Button.right)
                self.is_right_clicking = True
                self.click_cooldown = self.cooldown_frames
                return "Right Click", pinch_point
        elif self.is_right_clicking:
            mouse.release(Button.right)
            self.is_right_clicking = False
            self.click_cooldown = self.cooldown_frames

        # Ready indicators
        if thumb_index_distance < self.ready_threshold:
            ready_point = ((thumb_x + index_x) // 2, (thumb_y + index_y) // 2)
            cv2.circle(frame, ready_point, 15, (0, 255, 0), 2)
        if thumb_middle_distance < self.ready_threshold:
            ready_point = ((thumb_x + middle_x) // 2, (thumb_y + middle_y) // 2)
            cv2.circle(frame, ready_point, 15, (255, 0, 0), 2)

        return "", None