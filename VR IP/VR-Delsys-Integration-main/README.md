# VR Integration & Delsys Streaming Project

This project focuses on the integration of a VR Application (built in Unity) and Delsys sensor data collection. Our main application logic is located inside the `Python` folder, primarily driven by the `combined_master.py` script. This script handles connecting to the Delsys sensors and acts as a WebSocket server to receive synchronized hand tracking data and image frames from the VR headset.

---

## 🛠️ 1. Dependencies & Installation

To run the Python server, you need to install the necessary Python packages. 

1. Ensure you have Python installed.
2. Open your command prompt or terminal and navigate to the `Python` directory of this project:
   ```bash
   cd Python
   ```
3. Install the base requirements from the provided `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
4. Install the additional dependencies specifically required by `combined_master.py`:
   ```bash
   pip install websockets opencv-python numpy
   ```

*(Note: `combined_master.py` relies on `websockets` for network communication, `opencv-python` (`cv2`) for saving image frames, `numpy` for data manipulation, and the Delsys AeroPy packages which use `pythonnet`.)*

---

## 🔑 1.5. Delsys API Key & License

To connect to the Delsys hardware, you must configure your API key and license.
1. Open the file `Python/AeroPy/TrignoBase.py`.
2. Locate the variables `key` and `license` near the top of the script.
3. Replace `"YOUR_API_KEY_HERE"` and `"YOUR_LICENSE_XML_HERE"` with the exact key string and license XML string provided by Delsys.

---

## 🌐 2. Network & Unity Configuration

For the VR headset to communicate with your PC, they must be configured correctly on your local network.

1. Connect both your PC and the VR headset to the **same Wi-Fi network**.
2. Open a Command Prompt on your PC and run:
   ```cmd
   ipconfig
   ```
3. Look for your network adapter and note down your **IPv4 Address** (e.g., `192.168.1.100`).
4. **Unity Configuration:** Open your Unity project, and insert this IP address into the connection settings of your VR application. 
5. Ensure the connection port in Unity is explicitly set to **`9002`**.
6. Build the Unity project and install the app onto your VR headset.

---

## 🚀 3. Running the System

**CRITICAL ORDER OF OPERATIONS:** You **MUST** start the Python server first before launching the game in VR. If the server is not running, the VR app will fail to connect.

### Step A: Start the Server (PC)
1. Open a terminal and navigate to the `Python` folder.
2. Run the `combined_master.py` script:
   ```bash
   python combined_master.py
   ```
3. The terminal will display messages indicating that the server is ready, Delsys is connecting, and the VR server is listening on port `9002`.

### Step B: Start the VR App
1. Put on your VR headset.
2. Launch the installed VR game/application.
3. The VR app will automatically connect to your PC. You should see a `[VR] Client Connected!` message appear in your Python terminal.
4. The system is now actively streaming and recording data. Hand data, Delsys EMG data, and frames will be saved automatically inside the `Python/dataset/Session_YYYY-MM-DD_HH-MM-SS/` directory.

### Step C: Stop the Recording
- To stop the server and ensure all data is safely saved, go to the terminal running the Python script and press **`Ctrl+C`**.
- Wait for the script to print `=== SESSION COMPLETE ===` to guarantee the CSV files have finished writing and the Delsys streaming has gracefully disconnected.
