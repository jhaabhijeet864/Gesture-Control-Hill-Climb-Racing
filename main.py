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
from overlay import draw_hud, draw_status, draw_calibration_panel, ModernOverlay
from plugins.plugin_manager import PluginManager


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
    
    # Initialize plugin system
    log.info("Initializing Game Glide with plugin system...")
    plugin_manager = PluginManager("config.yaml")
    
    # Get configuration from plugin system
    config = plugin_manager.config
    
    # Legacy compatibility - still support old direct controls
    keyboard = KeyboardController()
    mouse = MouseController()
    GAS_KEY = key_from_name(config.get_config("keys.gas", "right"))
    BRAKE_KEY = key_from_name(config.get_config("keys.brake", "left"))
    screen_width, screen_height = pyautogui.size()

    # Inference layer
    infer = HandInference(
        min_detection_confidence=config.get_config("detection.min_detection_confidence", 0.7),
        min_tracking_confidence=config.get_config("detection.min_tracking_confidence", 0.7),
        max_num_hands=config.get_config("detection.max_num_hands", 2),
    )
    mp_hands = infer.mp_hands

    # Capture layer
    vc = VideoCapture(0)
    frame_width, frame_height = vc.get_size()

    # Legacy tracking & mapping state (for backward compatibility)
    smoothing = int(config.get_config("ui.smoothing", 8))
    game_mode = True
    is_gas_pressed = False
    is_brake_pressed = False
    prev_cursor_x, prev_cursor_y = 0, 0
    
    # Legacy gesture classification (fallback)
    clicker = ClickDetector(
        pinch_threshold=config.get_config("gestures.pinch_threshold", 0.05),
        ready_threshold=config.get_config("gestures.ready_threshold", 0.1),
        cooldown_frames=config.get_config("gestures.click_cooldown_frames", 5),
    )

    # UI overlay state
    fps = 0.0
    last_time = time.time()
    calibration_mode = False
    calib_closed = None
    calib_open = None
    
    # Plugin system mode toggle
    use_plugin_system = True
    
    # Initialize modern overlay
    modern_overlay = ModernOverlay()
    show_hud = True  # Toggle HUD visibility
    
    # Show plugin system status
    status = plugin_manager.get_status()
    log.info(f"Plugin System Status: {status}")

    log.info("Starting capture. Press 'q' to quit, 'm' to toggle mode, 'c' to calibrate, 'p' to toggle plugin system, 'h' to toggle HUD.")
    
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

            # UI overlay - Modern HUD
            if show_hud:
                mode_indicator = "PLUGIN" if use_plugin_system else ("GAME" if game_mode else "CURSOR")
                mode_text = f"{mode_indicator} MODE"
                mode_color = (0, 255, 255) if use_plugin_system else ((0, 0, 255) if game_mode else (0, 255, 0))
                
                # Use modern overlay system
                modern_overlay.draw_main_hud(frame, mode_text, mode_color, fps, 
                                           clicker.pinch_threshold if not use_plugin_system else None,
                                           smoothing if not use_plugin_system else None, 
                                           plugin_mode=use_plugin_system)

            # Tracking - hand assignment
            left_hand = None
            right_hand = None
            detected_gestures = []
            active_actions = []

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
                    
                    # Enhanced landmark visualization
                    if config.get_config("ui.show_landmarks", True):
                        # Convert landmarks to list format for enhanced drawing
                        landmarks_list = []
                        for landmark in lm.landmark:
                            landmarks_list.append([landmark.x, landmark.y, landmark.z])
                        
                        modern_overlay.draw_hand_landmarks_enhanced(frame, landmarks_list, 
                                                                  hand_type=label)
                    else:
                        # Fallback to simple drawing
                        infer.draw(frame, lm)

            status_text = ""

            if use_plugin_system:
                # NEW: Plugin system gesture processing
                try:
                    # Process gestures through plugin system
                    gesture_results = plugin_manager.process_gestures(left_hand, right_hand, frame)
                    
                    # Execute actions based on detected gestures
                    execution_results = plugin_manager.execute_gestures(gesture_results)
                    
                    # Collect gesture data for visualization
                    for gesture in gesture_results:
                        gesture_info = {
                            'name': gesture.name,
                            'confidence': gesture.confidence,
                            'hand': gesture.hand,
                            'features': getattr(gesture, 'features', {})
                        }
                        detected_gestures.append(gesture_info)
                        
                        # Add to overlay history for trail effect
                        modern_overlay.add_gesture_to_history(gesture.name, gesture.confidence)
                    
                    # Collect active actions for display
                    for gesture_name, result in execution_results.items():
                        if result.get("success", False):
                            action_type = result.get("action_type", "")
                            action_desc = f"{action_type}: {result.get('action_params', {}).get('key', 'unknown')}"
                            active_actions.append(action_desc)
                    
                    # Set status text based on detected gestures
                    if detected_gestures:
                        active_gesture_names = [g['name'] for g in detected_gestures if g['confidence'] > 0.5]
                        status_text = " | ".join(active_gesture_names) if active_gesture_names else ""
                    
                    # Check for profile updates
                    updated_profiles = plugin_manager.reload_profiles()
                    if updated_profiles:
                        log.info(f"Reloaded profiles: {updated_profiles}")
                        
                except Exception as e:
                    log.error(f"Plugin system error: {e}")
                    status_text = "Plugin Error"
            
            else:
                # LEGACY: Original gesture processing for backward compatibility
                # ... (existing gesture logic for fist detection and mouse control)
                
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

            # UI overlay - Modern gesture status display
            if show_hud and (detected_gestures or active_actions):
                modern_overlay.draw_gesture_status(frame, detected_gestures, active_actions)
            
            # UI overlay - Legacy status (fallback)
            if not show_hud or (not detected_gestures and not active_actions):
                draw_status(frame, status_text, frame_height)
            
            # Show current profile in plugin mode
            if use_plugin_system and show_hud:
                active_profile = plugin_manager.profile_manager.get_active_profile()
                if active_profile:
                    # Use modern text display
                    modern_overlay.draw_text_with_background(
                        frame, f"Profile: {active_profile.name}", 
                        (10, frame_height - 60), modern_overlay.font_medium, 0.6,
                        modern_overlay.colors['white'], modern_overlay.colors['dark_bg'])

            cv2.imshow("Game Glide - Hand Gesture Control", frame)

            # Input handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('p'):
                use_plugin_system = not use_plugin_system
                log.info("Plugin System %s", "Activated" if use_plugin_system else "Deactivated")
                # Clean up legacy state when switching
                if use_plugin_system:
                    if is_gas_pressed:
                        keyboard.release(GAS_KEY)
                        is_gas_pressed = False
                    if is_brake_pressed:
                        keyboard.release(BRAKE_KEY)
                        is_brake_pressed = False
                    if clicker.is_left_clicking:
                        mouse.release(Button.left)
                        clicker.is_left_clicking = False
                    if clicker.is_right_clicking:
                        mouse.release(Button.right)
                        clicker.is_right_clicking = False
            if key == ord('m') and not use_plugin_system:
                game_mode = not game_mode
                log.info("Game Mode %s", "Activated" if game_mode else "Deactivated")
                if game_mode:
                    if clicker.is_left_clicking:
                        mouse.release(Button.left)
                        clicker.is_left_clicking = False
                    if clicker.is_right_clicking:
                        mouse.release(Button.right)
                        clicker.is_right_clicking = False
            
            if key == ord('h'):
                show_hud = not show_hud
                log.info("HUD %s", "Visible" if show_hud else "Hidden")
            
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
                    config.set_config("gestures.pinch_threshold", new_thr)
                    log.info("Saved pinch_threshold=%.4f", new_thr)
                    calibration_mode = False
                if key == 27:  # ESC
                    calibration_mode = False

            # Live tuning (only in legacy mode)
            if not use_plugin_system:
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
            plugin_manager.cleanup()
            
            # Legacy cleanup
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
