import cv2
import mediapipe as mp
from pynput.keyboard import Controller as KeyboardController, Key
from pynput.mouse import Button, Controller as MouseController
from time import sleep
import math
import numpy as np
import pyautogui

# Controllers for keyboard and mouse
keyboard = KeyboardController()
mouse = MouseController()

# Get screen size for mouse movement mapping
screen_width, screen_height = pyautogui.size()

# MediaPipe hand tracking solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    min_detection_confidence=0.7, 
    min_tracking_confidence=0.7,
    max_num_hands=2
)
mp_drawing = mp.solutions.drawing_utils

# Start the webcam
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Variable initialization
is_gas_pressed = False
is_brake_pressed = False
is_left_clicking = False
is_right_clicking = False
click_cooldown = 0
smoothing = 8  # Higher value = smoother cursor movement but more lag
prev_cursor_x, prev_cursor_y = 0, 0
left_hand_landmarks = None
right_hand_landmarks = None

# Game mode toggle - True means cursor control is disabled (game mode active)
game_mode = True  # Start with game mode activated by default

# Hand fist (closed hand) detection function
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

# Improved click detection using pinch gesture (thumb and index finger)
def detect_click(landmarks, frame):
    global is_left_clicking, is_right_clicking, click_cooldown, game_mode
    
    # Skip click detection in game mode
    if game_mode:
        return "", None
        
    if click_cooldown > 0:
        click_cooldown -= 1
        return "", None

    # Get landmark positions
    thumb_tip = landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    
    # Calculate distances for pinch detection
    thumb_index_distance = math.sqrt(
        (thumb_tip.x - index_tip.x)**2 + 
        (thumb_tip.y - index_tip.y)**2)
    
    thumb_middle_distance = math.sqrt(
        (thumb_tip.x - middle_tip.x)**2 + 
        (thumb_tip.y - middle_tip.y)**2)
    
    # Convert to pixel coordinates for visualization
    thumb_x, thumb_y = int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0])
    index_x, index_y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])
    middle_x, middle_y = int(middle_tip.x * frame.shape[1]), int(middle_tip.y * frame.shape[0])
    
    pinch_threshold = 0.05  # Adjust threshold based on your preference
    
    # Left click - thumb and index finger pinch
    if thumb_index_distance < pinch_threshold:
        # Draw pinch circle for visual feedback
        pinch_point = ((thumb_x + index_x) // 2, (thumb_y + index_y) // 2)
        cv2.circle(frame, pinch_point, 10, (0, 255, 0), -1)  # Green circle for left click
        
        if not is_left_clicking and click_cooldown == 0:
            mouse.press(Button.left)
            is_left_clicking = True
            click_cooldown = 5  # Reduced cooldown for more responsiveness
            return "Left Click", pinch_point
    elif is_left_clicking:
        mouse.release(Button.left)
        is_left_clicking = False
        click_cooldown = 5
    
    # Right click - thumb and middle finger pinch
    if thumb_middle_distance < pinch_threshold:
        # Draw pinch circle for visual feedback
        pinch_point = ((thumb_x + middle_x) // 2, (thumb_y + middle_y) // 2)
        cv2.circle(frame, pinch_point, 10, (255, 0, 0), -1)  # Blue circle for right click
        
        if not is_right_clicking and click_cooldown == 0:
            mouse.press(Button.right)
            is_right_clicking = True
            click_cooldown = 5
            return "Right Click", pinch_point
    elif is_right_clicking:
        mouse.release(Button.right)
        is_right_clicking = False
        click_cooldown = 5
    
    # Draw "ready to click" indicators when fingers are close
    if thumb_index_distance < 0.1:  # Larger threshold for "ready" indication
        ready_point = ((thumb_x + index_x) // 2, (thumb_y + index_y) // 2)
        cv2.circle(frame, ready_point, 15, (0, 255, 0), 2)  # Green outline for left click ready
    
    if thumb_middle_distance < 0.1:  # Larger threshold for "ready" indication
        ready_point = ((thumb_x + middle_x) // 2, (thumb_y + middle_y) // 2)
        cv2.circle(frame, ready_point, 15, (255, 0, 0), 2)  # Blue outline for right click ready
    
    return "", None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the video for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for hand detection
    results = hands.process(rgb_frame)

    # Reset hand landmarks
    left_hand_landmarks = None
    right_hand_landmarks = None
    
    # Display the current mode status
    mode_text = "GAME MODE: Cursor Control OFF" if game_mode else "CURSOR MODE: Cursor Control ON"
    mode_color = (0, 0, 255) if game_mode else (0, 255, 0)
    cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
    cv2.putText(frame, "Press 'M' to toggle mode", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # If hands are detected
    if results.multi_hand_landmarks:
        # Identify left and right hands
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Get handedness (left or right hand)
            handedness = results.multi_handedness[hand_idx].classification[0].label
            
            if handedness == "Left":  # This is actually the right hand due to mirroring
                right_hand_landmarks = hand_landmarks
            else:  # This is actually the left hand due to mirroring
                left_hand_landmarks = hand_landmarks
                
            # Draw the hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Color the landmarks with different colors
            for idx, landmark in enumerate(hand_landmarks.landmark):
                # Get coordinates
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                
                # Different colors for each finger's tip
                if idx == mp_hands.HandLandmark.THUMB_TIP:
                    color = (0, 0, 255)  # Red for Thumb
                elif idx == mp_hands.HandLandmark.INDEX_FINGER_TIP:
                    color = (0, 255, 0)  # Green for Index Finger
                elif idx == mp_hands.HandLandmark.MIDDLE_FINGER_TIP:
                    color = (255, 0, 0)  # Blue for Middle Finger
                elif idx == mp_hands.HandLandmark.RING_FINGER_TIP:
                    color = (0, 255, 255)  # Yellow for Ring Finger
                elif idx == mp_hands.HandLandmark.PINKY_TIP:
                    color = (255, 0, 255)  # Purple for Pinky Finger
                else:
                    color = (255, 255, 255)  # White for others

                # Draw a circle on each landmark
                cv2.circle(frame, (x, y), 5, color, -1)
        
        # Handle right hand for game control (gas/brake)
        status_text = ""
        if right_hand_landmarks:
            # Check if fist gesture is detected for gas/brake
            if is_fist(right_hand_landmarks):
                if not is_gas_pressed:
                    print("Right hand closed: Accelerator (Gas)")
                    keyboard.press(Key.right)
                    is_gas_pressed = True
                    is_brake_pressed = False
                status_text = "Accelerator (Gas)"
            else:
                if is_gas_pressed:
                    keyboard.release(Key.right)
                    is_gas_pressed = False
                
            # Check for click gesture with right hand (only if not in game mode)
            if not game_mode:
                click_status, click_point = detect_click(right_hand_landmarks, frame)
                if click_status:
                    status_text += f" | {click_status}"
        
        # Handle left hand for mouse cursor control (only if not in game mode)
        if left_hand_landmarks:
            # Check if fist gesture is detected for brake (always active, regardless of mode)
            if is_fist(left_hand_landmarks):
                if not is_brake_pressed:
                    print("Left hand closed: Brake")
                    keyboard.press(Key.left)
                    is_brake_pressed = True
                status_text += " | Brake"
            else:
                if is_brake_pressed:
                    keyboard.release(Key.left)
                    is_brake_pressed = False
            
            # Only handle cursor control if not in game mode
            if not game_mode:
                # Use index finger tip for cursor position
                index_tip = left_hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                
                # Map the coordinates from camera space to screen space with scaling
                cursor_x = int(index_tip.x * screen_width)
                cursor_y = int(index_tip.y * screen_height)
                
                # Smooth mouse movement
                if prev_cursor_x == 0:
                    prev_cursor_x, prev_cursor_y = cursor_x, cursor_y
                else:
                    cursor_x = prev_cursor_x + (cursor_x - prev_cursor_x) // smoothing
                    cursor_y = prev_cursor_y + (cursor_y - prev_cursor_y) // smoothing
                    prev_cursor_x, prev_cursor_y = cursor_x, cursor_y
                
                # Move mouse to index finger position
                mouse.position = (cursor_x, cursor_y)
                
                # Display the cursor position on screen
                cv2.putText(frame, f"Cursor: ({cursor_x}, {cursor_y})", 
                           (10, frame_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (255, 255, 255), 2)
                
                # Check for click gesture with left hand
                click_status, click_point = detect_click(left_hand_landmarks, frame)
                if click_status:
                    status_text += f" | {click_status}"
        
        # Display the status text on the frame if an action is detected
        if status_text:
            cv2.putText(frame, status_text, (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame with hand gesture control
    cv2.imshow("Hand Gesture Control", frame)

    # Key handling
    key = cv2.waitKey(1) & 0xFF
    
    # Exit the loop when 'q' is pressed
    if key == ord('q'):
        break
    
    # Toggle game mode when 'm' is pressed
    if key == ord('m'):
        game_mode = not game_mode
        print(f"Game Mode {'Activated' if game_mode else 'Deactivated'}")
        
        # If switching to game mode, make sure to release any mouse buttons
        if game_mode and (is_left_clicking or is_right_clicking):
            if is_left_clicking:
                mouse.release(Button.left)
                is_left_clicking = False
            if is_right_clicking:
                mouse.release(Button.right)
                is_right_clicking = False

# Release resources
if is_gas_pressed:
    keyboard.release(Key.right)
if is_brake_pressed:
    keyboard.release(Key.left)
if is_left_clicking:
    mouse.release(Button.left)
if is_right_clicking:
    mouse.release(Button.right)

cap.release()
cv2.destroyAllWindows()