import cv2


def draw_hud(frame, mode_text: str, mode_color: tuple[int, int, int], fps: float, pinch_thr: float, smoothing: int):
    cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
    cv2.putText(frame, f"FPS: {fps:4.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    cv2.putText(frame, f"PinchThr: {pinch_thr:.3f}  Smooth: {smoothing}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, "Keys: m=toggle mode, q=quit, c=calibrate, [/] pinch-,+, -/= smooth-+,",
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)


def draw_status(frame, text: str, frame_height: int):
    if not text:
        return
    cv2.putText(frame, text, (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, "", (10, frame_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def draw_calibration_panel(frame, frame_width: int):
    cv2.rectangle(frame, (10, 140), (frame_width - 10, 240), (30, 30, 30), -1)
    cv2.putText(frame, "CALIBRATION MODE:", (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, "1: sample PINCH CLOSED (thumb-index touching)", (20, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, "2: sample PINCH OPEN (thumb-index apart)", (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(frame, "s: save & exit  /  esc: exit w/o save", (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
