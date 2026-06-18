import asyncio
import websockets
import struct
import numpy as np
import cv2
import os
import csv
import sys
import time
import threading
from datetime import datetime

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
# --- NETWORK (VR) ---
IP = "0.0.0.0"
PORT = 9002
SAVE_DIR = "dataset"

# --- DELSYS SENSORS ---
# Mode: EMG + Accelerometer + Gyroscope
DELSYS_MODE = "EMG raw (1259 Hz), skin check (74 Hz), ACC 2g (148 Hz), GYRO 250 dps (148 Hz)"
# Duration: Set to None to run until Ctrl+C, or a number (e.g. 60) to stop auto.
RECORD_DURATION = None 

# ==============================================================================
# 2. SESSION SETUP (Global State)
# ==============================================================================
# Create a unique session ID for THIS run
session_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
session_path = os.path.join(SAVE_DIR, f"Session_{session_id}")
frames_dir = os.path.join(session_path, "frames")
hands_csv_path = os.path.join(session_path, "vr_hands.csv")

# Ensure folders exist
os.makedirs(frames_dir, exist_ok=True)

# ==============================================================================
# 3. DELSYS SYSTEM SETUP
# ==============================================================================
current_dir = os.getcwd()
if current_dir not in sys.path: sys.path.append(current_dir)

try:
    from AeroPy.DataManager import DataKernel 
    from AeroPy.TrignoBase import TrignoBase
except ImportError:
    print("[Error] Script must be in 'Example-Applications-main/Python/'")
    sys.exit(1)

class MockController:
    def __init__(self, trigno_base):
        self.trigno_base = trigno_base
        self.DataHandler = DataKernel(trigno_base)
        self.streamYTData = False
        self.pauseFlag = False
        self.EMGplot = None
    def threadManager(self, start, stop): pass

class DelsysRecorder:
    def __init__(self, output_folder, mode):
        self.output_folder = output_folder
        self.target_mode = mode
        self.base = TrignoBase(None)
        self.base.collection_data_handler = MockController(self.base)
        self.sensor_map = [] 
        self.stop_event = threading.Event()

    def run(self):
        print("[Delsys] Connecting...")
        self.base.Connect_Callback()
        self.base.Scan_Callback()
        
        # Configure Mode
        sensor_count = self.base.SensorCount
        if sensor_count == 0:
            print("[Delsys] Error: No sensors found.")
            return

        for i in range(sensor_count):
            available = self.base.getSampleModes(i)
            found = next((m for m in available if self.target_mode.upper() in m.upper()), None)
            
            if found:
                self.base.setSampleMode(i, found)
                print(f"[Delsys] Sensor {i} set to: {found[:30]}...")
            else:
                print(f"[Delsys] Sensor {i}: Mode not found, using default.")

            # Map Channels
            updated_sensor = self.base.TrigBase.GetSensorObject(i)
            for ch in updated_sensor.TrignoChannels:
                if "SkinCheck" in str(ch.Type) or "IMP" in ch.Name: continue
                self.sensor_map.append({
                    'name': f"S{updated_sensor.PairNumber}_{ch.Name.replace(' ', '_')}",
                    'rate': ch.SampleRate
                })

        print("[Delsys] Starting Stream...")
        self.base.Start_Callback(start_trigger=False, stop_trigger=False)
        
        start_time = time.time()
        dummy_queue = []
        
        # Recording Loop
        try:
            while not self.stop_event.is_set():
                self.base.collection_data_handler.DataHandler.processData(dummy_queue)
                time.sleep(0.002) 
        except Exception as e:
            print(f"[Delsys] Error: {e}")
        finally:
            print("[Delsys] Stopping...")
            self.base.Stop_Callback()
            self.save_csv(start_time)

    def save_csv(self, start_time):
        filename = os.path.join(self.output_folder, "delsys_emg.csv")
        print(f"[Delsys] Saving to {filename}...")

        all_data = self.base.collection_data_handler.DataHandler.allcollectiondata
        if not all_data or len(all_data) == 0: return

        # Handle Header Mismatch Fallback
        if len(self.sensor_map) == len(all_data):
            headers = ["Rel_Time", "System_Time"] + [x['name'] for x in self.sensor_map]
            rates = [x['rate'] for x in self.sensor_map]
        else:
            headers = ["Rel_Time", "System_Time"] + [f"Ch_{i}" for i in range(len(all_data))]
            rates = [2000.0] * len(all_data)

        try:
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)

                # Master Channel Logic
                max_len = max(len(ch) for ch in all_data)
                master_idx = next(i for i, ch in enumerate(all_data) if len(ch) == max_len)
                master_rate = rates[master_idx] if master_idx < len(rates) else 2000.0

                # Trim Zeros
                start_idx = 0
                if len(all_data[master_idx]) > 0:
                    for i, val in enumerate(all_data[master_idx]):
                        if abs(val) > 1e-6:
                            start_idx = i
                            break

                for i in range(start_idx, max_len):
                    t_rel = (i - start_idx) / master_rate
                    # Sync Clock
                    t_clock = datetime.fromtimestamp(start_time + t_rel).strftime("%H:%M:%S.%f")[:-3]
                    
                    row = [f"{t_rel:.6f}", t_clock]
                    for ch_idx, col_data in enumerate(all_data):
                        target_idx = int(t_rel * (rates[ch_idx] if ch_idx < len(rates) else master_rate)) + start_idx
                        if target_idx < len(col_data):
                            row.append(f"{col_data[target_idx]:.6f}")
                        else:
                            row.append("")
                    writer.writerow(row)
            print("[Delsys] Save Complete.")
        except Exception as e:
            print(f"[Delsys] Save Failed: {e}")

# ==============================================================================
# 4. VR RECEIVER (Asyncio Handler)
# ==============================================================================
async def vr_handler(websocket):
    print("[VR] Client Connected!")
    
    # Initialize CSV for Hands
    if not os.path.exists(hands_csv_path):
        with open(hands_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ["Frame", "Timestamp", "System_Time"] # Added System_Time
            for i in range(25):
                header.extend([f"J{i}_Valid", f"J{i}_X", f"J{i}_Y", f"J{i}_Z", f"J{i}_Dist"])
            writer.writerow(header)

    try:
        async for message in websocket:
            current_clock = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            
            if len(message) < 32: continue

            # Parse Header
            header = struct.unpack('=I Q H H H H I I I', message[:32])
            frame_idx, timestamp, cw, ch, dw, dh, l_col, l_dep, l_hand = header

            # Data Offsets
            offset = 32
            hands_data = message[offset: offset + l_hand]
            offset += l_hand
            color_data = message[offset: offset + l_col]
            offset += l_col
            depth_data = message[offset: offset + l_dep]

            # Process Hands
            if l_hand > 0 and l_hand % 425 == 0:
                num_hands = l_hand // 425
                ptr = 0
                for h in range(num_hands):
                    hand_chunk = hands_data[ptr : ptr + 425]
                    ptr += 425
                    
                    row = [frame_idx, timestamp, current_clock] # Added Clock
                    chunk_ptr = 0
                    has_valid_data = False

                    for _ in range(25):
                        val = struct.unpack_from('=?ffff', hand_chunk, chunk_ptr)
                        chunk_ptr += 17
                        row.extend(val)
                        if val[0] == True: has_valid_data = True

                    if has_valid_data:
                        with open(hands_csv_path, 'a', newline='') as f:
                            csv.writer(f).writerow(row)

            # Save Images (Every 10th frame to save space/perf)
            if frame_idx % 10 == 0:
                print(f"[VR] Recv Frame {frame_idx} | HandBytes: {l_hand}", end='\r')
                
                # Color
                if cw > 0 and len(color_data) == (cw * ch * 4):
                    img = np.frombuffer(color_data, dtype=np.uint8).reshape(ch, cw, 4)
                    cv2.imwrite(f"{frames_dir}/color_{frame_idx}.png", cv2.flip(img, 0))

                # Depth
                if dw > 0 and len(depth_data) == (dw * dh * 4):
                    img = np.frombuffer(depth_data, dtype=np.uint8).reshape(dh, dw, 4)
                    cv2.imwrite(f"{frames_dir}/depth_{frame_idx}.png", cv2.flip(img[:,:,0], 0))

    except Exception as e:
        print(f"[VR] Error: {e}")
    finally:
        print("\n[VR] Client Disconnected")

# ==============================================================================
# 5. MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    print(f"--- COMBINED SERVER READY ---")
    print(f"Saving Session to: {session_path}")
    
    # 1. Start Delsys in a separate Thread
    delsys_recorder = DelsysRecorder(session_path, DELSYS_MODE)
    delsys_thread = threading.Thread(target=delsys_recorder.run)
    delsys_thread.start()
    
    # 2. Start VR Server (Asyncio - Runs on Main Thread)
    try:
        async def start_server():
            print(f"[VR] Listening on {IP}:{PORT}")
            async with websockets.serve(vr_handler, IP, PORT):
                await asyncio.Future() # Run forever

        asyncio.run(start_server())
        
    except KeyboardInterrupt:
        print("\n[Main] Stopping...")
    finally:
        # Stop Delsys
        delsys_recorder.stop_event.set()
        delsys_thread.join()
        print("=== SESSION COMPLETE ===")