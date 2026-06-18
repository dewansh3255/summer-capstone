import sys
import os
import time
import csv
from datetime import datetime

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# --- MODE SELECTION ---
# ENABLED: EMG + Accelerometer + Gyroscope
DESIRED_MODE = "EMG raw (1259 Hz), skin check (74 Hz), ACC 2g (148 Hz), GYRO 250 dps (148 Hz)"

# DISABLED: High-Speed EMG Only (Uncomment if you ONLY want muscle data)
# DESIRED_MODE = "EMG raw (2148 Hz), skin check (74 Hz), +/-5.5mV, 20-450Hz"

DURATION = 15

# ==============================================================================
# SETUP
# ==============================================================================
current_dir = os.getcwd()
if current_dir not in sys.path: sys.path.append(current_dir)

try:
    from AeroPy.DataManager import DataKernel 
    from AeroPy.TrignoBase import TrignoBase
except ImportError:
    sys.exit("[Error] Script must be in 'Example-Applications-main/Python/'")

class MockController:
    def __init__(self, trigno_base):
        self.trigno_base = trigno_base
        self.DataHandler = DataKernel(trigno_base)
        self.streamYTData = False
        self.pauseFlag = False
        self.EMGplot = None
    def threadManager(self, start, stop): pass

class EmgImuRecorder:
    def __init__(self, duration, mode):
        self.duration = duration
        self.target_mode = mode
        self.base = TrignoBase(None)
        self.base.collection_data_handler = MockController(self.base)
        self.sensor_map = [] 

    def run(self):
        print("--- DELSYS EMG + IMU RECORDER ---")
        
        # 1. CONNECT & SCAN
        print("[1/5] Connecting...")
        self.base.Connect_Callback()
        self.base.Scan_Callback()
        
        # 2. CONFIGURE MODE
        print(f"[2/5] Setting Mode: '{self.target_mode[:40]}...'")
        
        sensor_count = self.base.SensorCount
        if sensor_count == 0:
            print("[Error] No sensors found.")
            return

        for i in range(sensor_count):
            # Find closest matching mode
            available_modes = self.base.getSampleModes(i)
            found_mode = None
            for m in available_modes:
                if self.target_mode.upper() in m.upper():
                    found_mode = m
                    break
            
            if found_mode:
                self.base.setSampleMode(i, found_mode)
                print(f"  -> Sensor {i}: Mode Set Successfully")
            else:
                print(f"  -> Sensor {i}: [Warning] Mode not found, using default.")

            # MAP CHANNELS
            # Re-fetch sensor to get active channels
            updated_sensor = self.base.TrigBase.GetSensorObject(i)
            
            for ch in updated_sensor.TrignoChannels:
                # SKIP SKIN CHECK (Fixes the header mismatch)
                # Delsys does not stream SkinCheck data to the collector, 
                # so we must exclude it from our header map.
                if "SkinCheck" in str(ch.Type) or "IMP" in ch.Name:
                    continue

                clean_name = ch.Name.replace(' ', '_')
                col_name = f"S{updated_sensor.PairNumber}_{clean_name}"
                
                self.sensor_map.append({
                    'name': col_name,
                    'rate': ch.SampleRate
                })

        # 3. START STREAM
        print("[3/5] Starting Stream...")
        self.base.Start_Callback(start_trigger=False, stop_trigger=False)

        # 4. RECORD
        print(f"[4/5] Recording for {self.duration}s...")
        start_time = time.time()
        dummy_queue = []
        
        try:
            while (time.time() - start_time) < self.duration:
                self.base.collection_data_handler.DataHandler.processData(dummy_queue)
                time.sleep(0.005) # 5ms poll
        except KeyboardInterrupt:
            pass

        # 5. SAVE
        print("\n[5/5] Saving Data...")
        self.base.Stop_Callback()
        self.save_csv(start_time)

    def save_csv(self, start_time):
        time_str = datetime.fromtimestamp(start_time).strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"EmgImu_{time_str}.csv"
        print(f"      Writing to {filename}...")

        all_data = self.base.collection_data_handler.DataHandler.allcollectiondata
        
        if not all_data:
            print("[Error] No data collected.")
            return

        # Header logic
        if len(self.sensor_map) == len(all_data):
            headers = ["Rel_Time", "System_Time"] + [x['name'] for x in self.sensor_map]
            rates = [x['rate'] for x in self.sensor_map]
        else:
            print(f"      [Warn] Header Mismatch (Expected {len(self.sensor_map)}, Got {len(all_data)})")
            headers = ["Rel_Time", "System_Time"] + [f"Ch_{i}" for i in range(len(all_data))]
            rates = [2000.0] * len(all_data)

        try:
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)

                # Find Master Channel
                max_len = max(len(ch) for ch in all_data)
                master_idx = 0
                for i, ch in enumerate(all_data):
                    if len(ch) == max_len:
                        master_idx = i
                        break
                master_rate = rates[master_idx]

                # Trim Zeros
                start_idx = 0
                if len(all_data[master_idx]) > 0:
                    for i, val in enumerate(all_data[master_idx]):
                        if abs(val) > 1e-6:
                            start_idx = i
                            break

                # Write Rows
                for i in range(start_idx, max_len):
                    t_rel = (i - start_idx) / master_rate
                    
                    # Real Clock Time
                    abs_time = start_time + t_rel
                    t_clock = datetime.fromtimestamp(abs_time).strftime("%H:%M:%S.%f")[:-3]
                    
                    row = [f"{t_rel:.6f}", t_clock]
                    
                    for ch_idx, channel_data in enumerate(all_data):
                        this_rate = rates[ch_idx] if ch_idx < len(rates) else master_rate
                        target_idx = int(t_rel * this_rate) + start_idx
                        
                        if target_idx < len(channel_data):
                            row.append(f"{channel_data[target_idx]:.6f}")
                        else:
                            row.append("")
                    writer.writerow(row)
            print("[Success] Done.")
        except Exception as e:
            print(f"[Error] Save failed: {e}")

if __name__ == "__main__":
    app = EmgImuRecorder(DURATION, DESIRED_MODE)
    app.run()