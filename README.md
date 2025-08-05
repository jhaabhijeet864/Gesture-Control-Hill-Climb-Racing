
# 🚗💨 Gesture-Control Hill Climb Racing 👋👾

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![Gesture Control](https://img.shields.io/badge/Gesture%20Control-Enabled-orange?logo=handshake)](https://github.com/jhaabhijeet864/Gesture-Control-Hill-Climb-Racing)
[![Game](https://img.shields.io/badge/Game-Hill%20Climb%20Racing-green?logo=gamepad)]()
[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Open Source](https://img.shields.io/badge/Open%20Source-Yes-brightgreen?logo=github)]()
[![Stars](https://img.shields.io/github/stars/jhaabhijeet864/Gesture-Control-Hill-Climb-Racing?style=social)](https://github.com/jhaabhijeet864/Gesture-Control-Hill-Climb-Racing/stargazers)
[![Forks](https://img.shields.io/github/forks/jhaabhijeet864/Gesture-Control-Hill-Climb-Racing?style=social)](https://github.com/jhaabhijeet864/Gesture-Control-Hill-Climb-Racing/forks)
[![Issues](https://img.shields.io/github/issues/jhaabhijeet864/Gesture-Control-Hill-Climb-Racing?color=red)](https://github.com/jhaabhijeet864/Gesture-Control-Hill-Climb-Racing/issues)
[![Pull Requests](https://img.shields.io/github/issues-pr/jhaabhijeet864/Gesture-Control-Hill-Climb-Racing?color=blue)](https://github.com/jhaabhijeet864/Gesture-Control-Hill-Climb-Racing/pulls)
[![Last Commit](https://img.shields.io/github/last-commit/jhaabhijeet864/Gesture-Control-Hill-Climb-Racing?color=purple)](https://github.com/jhaabhijeet864/Gesture-Control-Hill-Climb-Racing/commits)

> 🚗💨 Experience Hill Climb Racing with our innovative hand gesture control system. Move your hand to accelerate, brake, and tilt, bringing a whole new level of fun and interactivity to your gameplay. 👋👾

---

## ✨ Features

- **🕹 Game Control:**
  - 💨 **Accelerate:** Make a fist with your **right** hand.
  - 🛑 **Brake:** Make a fist with your **left** hand.
- **🖱️ Mouse Control:**
  - 👆 **Move Cursor:** Move your **left hand's index finger**.
  - 🖱️ **Left Click:** Pinch your **thumb** and **index** finger together.
  - ➡️ **Right Click:** Pinch your **thumb** and **middle** finger together.
- **🔄 Game Mode Toggle:** Press `M` to switch between game and cursor modes.
- **👁️ Visual Feedback:** See hand landmarks and recognized gestures overlaid on the webcam feed. Mode status (Game Mode or Cursor Mode) is displayed for clarity.

---

## ⚙️ Requirements

- **Libraries:**
  ```bash
  pip install opencv-python mediapipe pynput pyautogui
  ```
- **Hardware:**
  - Standard webcam
  - Windows OS (required for direct keyboard input simulation)

---

## 🚀 How to Run

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/jhaabhijeet864/Gesture-Control-Hill-Climb-Racing.git
   cd Gesture-Control-Hill-Climb-Racing
   ```

2. **Install Required Python Libraries:**
   ```bash
   pip install opencv-python mediapipe pynput pyautogui
   ```

3. **Ensure Your Webcam is Connected**  
   The gesture control system requires a working webcam.

4. **Launch Hill Climb Racing Game**  
   Start the Hill Climb Racing game on your PC.

5. **Run the Gesture Control Script:**
   ```bash
   python main.py
   ```
   The script will open a webcam window and start recognizing your hand gestures.  
   - Make a fist with your right hand to accelerate.
   - Make a fist with your left hand to brake.
   - Use mouse gestures if you toggle to cursor mode (`M` key).

6. **Switch Between Game and Mouse Control Modes:**
   - Press the `M` key to toggle between controlling the game and controlling your mouse cursor.

7. **Quit the Program:**
   - Press `ESC` in the webcam window to exit the gesture control.

---

## 🧩 Project Structure

```
Gesture-Control-Hill-Climb-Racing/
│
├── main.py                # Main script for gesture control
├── README.md              # Project documentation
├── requirements.txt       # Required Python packages
├── LICENSE                # MIT License
└── (other source files)   # Supporting modules and assets
```

---

## 🛠️ Troubleshooting

- **Camera Not Detected:**  
  Ensure your webcam is connected and accessible.
- **Keyboard/Mouse Actions Not Working:**  
  The script is tested for Windows. For other OS, some features may require adaptation.
- **Dependencies Issues:**  
  Double-check that all libraries are installed. Use Python 3.8+.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!  
Feel free to check the [issues page](https://github.com/jhaabhijeet864/Gesture-Control-Hill-Climb-Racing/issues).

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgements

- [OpenCV](https://opencv.org/)
- [MediaPipe](https://mediapipe.dev/)
- [PyAutoGUI](https://pyautogui.readthedocs.io/)
- [Hill Climb Racing](https://www.fingersoft.com/games/hill-climb-racing/)

---

If you want to update your README.md file with this content, let me know!
