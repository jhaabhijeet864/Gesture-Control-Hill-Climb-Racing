import os
import time
import math
import logging
import cv2
import pyautogui
from pynput.keyboard import Controller as KeyboardController
from pynput.mouse import Button, Controller as MouseController

from gestures import is_fist, ClickDetector
from config_utils import load_config, save_config, key_from_name, CONFIG_PATH
from capture import VideoCapture
from inference import HandInference
from overlay import draw_hud, draw_status, draw_calibration_panel


# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(os.path.join("logs", "run.log"), encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("gesture-app")


def measure_thumb_index_distance(hand_landmarks, mp_hands):
    """Calculate distance between thumb and index finger tips."""
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    return math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y)


def main():
    """Main orchestrator: capture → inference → gesture classification → mapping → actuation → overlay."""
    cfg = load_config()

    # Actuation layer
    keyboard = KeyboardController()
    mouse = MouseController()
    GAS_KEY = key_from_name(cfg["keys"].get("gas"))
    BRAKE_KEY = key_from_name(cfg["keys"].get("brake"))
    screen_width, screen_height = pyautogui.size()

    # Inference layer
    infer = HandInference(
        min_detection_confidence=cfg["min_detection_confidence"],
        min_tracking_confidence=cfg["min_tracking_confidence"],
        max_num_hands=cfg["max_num_hands"],
    )
    mp_hands = infer.mp_hands

    # Capture layer
    vc = VideoCapture(0)
    frame_width, frame_height = vc.get_size()

    # Tracking & mapping state
    smoothing = int(cfg["smoothing"])
    game_mode = True
    is_gas_pressed = False
    is_brake_pressed = False
    prev_cursor_x, prev_cursor_y = 0, 0
    
    # Gesture classification
    clicker = ClickDetector(
        pinch_threshold=cfg["pinch_threshold"],
        ready_threshold=cfg["ready_threshold"],
        cooldown_frames=cfg["click_cooldown_frames"],
    )

    # UI overlay state
    fps = 0.0
    last_time = time.time()
    calibration_mode = False
    calib_closed = None
    calib_open = None

    log.info("Starting capture. Press 'q' to quit, 'm' to toggle mode, 'c' to calibrate.")
    
    try:
        while True:
            # Capture
            ret, frame = vc.read()
            if not ret:
                log.error("Failed to read frame from camera.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Inference
            results = infer.process(rgb)

            # UI overlay - FPS tracking
            now = time.time()
            dt = max(1e-6, now - last_time)
            inst = 1.0 / dt
            fps = 0.9 * fps + 0.1 * inst if fps > 0 else inst
            last_time = now

            # UI overlay - HUD
            mode_text = "GAME MODE: Cursor OFF" if game_mode else "CURSOR MODE: Cursor ON"
            mode_color = (0, 0, 255) if game_mode else (0, 255, 0)
            draw_hud(frame, mode_text, mode_color, fps, clicker.pinch_threshold, smoothing)

            # Tracking - hand assignment
            left_hand = None
            right_hand = None

            if results.multi_hand_landmarks:
                for idx, lm in enumerate(results.multi_hand_landmarks):
                    # Mirror handedness for webcam flip
                    if results.multi_handedness and idx < len(results.multi_handedness):
                        label = results.multi_handedness[idx].classification[0].label
                    else:
                        label = "Right"
                    
                    if label == "Left":
                        right_hand = lm  # Flipped
                    else:
                        left_hand = lm
                    
                    infer.draw(frame, lm)

            status_text = ""

            # Gesture classification & mapping - Right hand (Gas)
            if right_hand:
                if is_fist(right_hand):
                    if not is_gas_pressed:
                        # Actuation
                        keyboard.press(GAS_KEY)
                        is_gas_pressed = True
                        if is_brake_pressed:
                            keyboard.release(BRAKE_KEY)
                            is_brake_pressed = False
                    status_text = "Accelerator (Gas)"
                else:
                    if is_gas_pressed:
                        keyboard.release(GAS_KEY)
                        is_gas_pressed = False

                if not game_mode:
                    click_status, _ = clicker.detect_click(right_hand, frame, mouse, Button, game_mode)
                    if click_status:
                        status_text += (" | " if status_text else "") + click_status

            # Gesture classification & mapping - Left hand (Brake + Cursor)
            if left_hand:
                if is_fist(left_hand):
                    if not is_brake_pressed:
                        keyboard.press(BRAKE_KEY)
                        is_brake_pressed = True
                    status_text += (" | " if status_text else "") + "Brake"
                else:
                    if is_brake_pressed:
                        keyboard.release(BRAKE_KEY)
                        is_brake_pressed = False

                if not game_mode:
                    # Tracking - cursor smoothing
                    idx_tip = left_hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    cx = int(idx_tip.x * screen_width)
                    cy = int(idx_tip.y * screen_height)
                    if prev_cursor_x == 0:
                        prev_cursor_x, prev_cursor_y = cx, cy
                    else:
                        cx = prev_cursor_x + (cx - prev_cursor_x) // max(1, smoothing)
                        cy = prev_cursor_y + (cy - prev_cursor_y) // max(1, smoothing)
                        prev_cursor_x, prev_cursor_y = cx, cy
                    
                    # Actuation
                    mouse.position = (cx, cy)
                    click_status, _ = clicker.detect_click(left_hand, frame, mouse, Button, game_mode)
                    if click_status:
                        status_text += (" | " if status_text else "") + click_status

            # UI overlay - calibration panel
            if calibration_mode:
                draw_calibration_panel(frame, frame_width)
                if left_hand or right_hand:
                    lm = left_hand or right_hand
                    dist = measure_thumb_index_distance(lm, mp_hands)
                    cv2.putText(frame, f"current dist: {dist:.4f}", (20, 255), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 180), 1)
                    cv2.putText(frame, f"closed={calib_closed} open={calib_open}", (20, 275), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 180), 1)

            # UI overlay - status
            draw_status(frame, status_text, frame_height)

            cv2.imshow("Hand Gesture Control", frame)

            # Input handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('m'):
                game_mode = not game_mode
                log.info("Game Mode %s", "Activated" if game_mode else "Deactivated")
                if game_mode:
                    if clicker.is_left_clicking:
                        mouse.release(Button.left)
                        clicker.is_left_clicking = False
                    if clicker.is_right_clicking:
                        mouse.release(Button.right)
                        clicker.is_right_clicking = False
            
            if key == ord('c'):
                calibration_mode = not calibration_mode
                calib_closed = None
                calib_open = None
                log.info("Calibration mode %s", "ON" if calibration_mode else "OFF")
            
            if calibration_mode:
                if key == ord('1') and (left_hand or right_hand):
                    lm = left_hand or right_hand
                    calib_closed = round(measure_thumb_index_distance(lm, mp_hands), 4)
                    log.info("Sampled CLOSED distance: %s", calib_closed)
                if key == ord('2') and (left_hand or right_hand):
                    lm = left_hand or right_hand
                    calib_open = round(measure_thumb_index_distance(lm, mp_hands), 4)
                    log.info("Sampled OPEN distance: %s", calib_open)
                if key == ord('s') and calib_closed and calib_open and calib_open > calib_closed:
                    new_thr = calib_closed + 0.35 * (calib_open - calib_closed)
                    clicker.set_thresholds(pinch_threshold=new_thr)
                    cfg["pinch_threshold"] = new_thr
                    save_config(cfg, CONFIG_PATH)
                    log.info("Saved pinch_threshold=%.4f to %s", new_thr, CONFIG_PATH)
                    calibration_mode = False
                if key == 27:  # ESC
                    calibration_mode = False

            # Live tuning
            if key == ord('['):
                clicker.set_thresholds(pinch_threshold=max(0.005, clicker.pinch_threshold - 0.005))
            if key == ord(']'):
                clicker.set_thresholds(pinch_threshold=min(0.200, clicker.pinch_threshold + 0.005))
            if key == ord('-'):
                smoothing = max(1, smoothing - 1)
            if key == ord('=') or key == ord('+'):
                smoothing = min(32, smoothing + 1)

    except Exception as e:
        log.exception("Unhandled exception: %s", e)
    finally:
        # Safe cleanup
        try:
            if is_gas_pressed:
                keyboard.release(GAS_KEY)
            if is_brake_pressed:
                keyboard.release(BRAKE_KEY)
            if clicker.is_left_clicking:
                mouse.release(Button.left)
            if clicker.is_right_clicking:
                mouse.release(Button.right)
        except Exception:
            pass
        vc.release()
        cv2.destroyAllWindows()
        log.info("Shutdown complete.")


if __name__ == "__main__":
    main()
