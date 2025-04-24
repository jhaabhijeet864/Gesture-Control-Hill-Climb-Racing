# 🎮 Hill Climb Racing Gesture Control

👋 Control the classic game Hill Climb Racing and even your mouse cursor using just your hands! This innovative project leverages the power of computer vision and hand tracking to bring you a unique and intuitive control experience. Using your webcam, it recognizes your gestures in real-time, translating them into game actions and mouse movements.

✨ **Features** ✨

* **🚗 Game Control:**
    * 💨 **Accelerate:** Make a fist with your **right** hand.
    * 🛑 **Brake:** Make a fist with your **left** hand.
* 🖱️ **Mouse Control:**
    * 👆 **Move Cursor:** Gently move your **left** hand's index finger.
    * 🖱️ **Left Click:** Pinch your **thumb** and **index** finger together.
    * ➡️ **Right Click:** Pinch your **thumb** and **middle** finger together.
* 🕹️ **Game Mode Toggle:** Press the `M` key on your keyboard to seamlessly switch between controlling the game and controlling your mouse cursor.
* 👁️ **Visual Feedback:** See your hand landmarks and recognized gestures overlaid on the live webcam feed. The current mode (Game Mode or Cursor Mode) is also displayed on the screen for clear indication.

**⚙️ Requirements**

* **Libraries:**
    ```bash
    pip install opencv-python mediapipe pynput pyautogui
    ```
* **Hardware:**
    * A standard webcam to capture your hand movements.
    * A Windows operating system (required for the `directkeys.py` script which utilizes the Windows API for direct keyboard input simulation).

**🚀 How to Run**

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/jhaabhijeet864/Gesture-Control-Hill-Climb-Racing.git](https://github.com/jhaabhijeet864/Gesture-Control-Hill-Climb-Racing.git)
    ```
2.  **Navigate to the Project Directory:**
    ```bash
    cd Gesture-Control-Hill-Climb-Racing
    ```
3.  **Run the Main Script:**
    ```bash
    python main.py
    ```
4.  **Position your webcam** so that it clearly captures your hands.

**🕹️ Usage**

* **Game Mode:**
    * ✊ Close your **right hand** into a fist to **accelerate** your vehicle.
    * ✊ Close your **left hand** into a fist to engage the **brakes**.
* **Cursor Mode:**
    * ☝️ Move your **left hand's index finger** to move the mouse cursor on the screen.
    * 🤏 Pinch your **thumb and index finger** together to perform a **left-click**.
    * 🤏 Pinch your **thumb and middle finger** together to perform a **right-click**.
* **Toggle Mode:** Press the `M` key on your keyboard to switch between **Game Mode** and **Cursor Mode**.
* **Quit:** Press the `Q` key to exit the program.

**📂 File Structure**





