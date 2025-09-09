import mediapipe as mp


class HandInference:
    def __init__(self, min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=2):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            max_num_hands=max_num_hands,
        )
        self.drawing = mp.solutions.drawing_utils

    def process(self, rgb_frame):
        return self.hands.process(rgb_frame)

    def draw(self, frame, hand_landmarks):
        self.drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
