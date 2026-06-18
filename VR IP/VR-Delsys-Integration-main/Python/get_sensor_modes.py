import sys
import os

# --- 1. SETUP PATHS ---
current_dir = os.getcwd()
if current_dir not in sys.path: sys.path.append(current_dir)

try:
    from AeroPy.TrignoBase import TrignoBase
except ImportError:
    sys.exit("[Error] Run this from 'Example-Applications-main/Python/'")

# --- 2. DUMMY HANDLER ---
# TrignoBase needs this to initialize without crashing
class DummyHandler:
    def __init__(self):
        self.streamYTData = False
        self.pauseFlag = False
        self.EMGplot = None
        self.DataHandler = self
        self.allcollectiondata = []
    def threadManager(self, start, stop): pass

# --- 3. MAIN SCRIPT ---
def list_modes():
    print("--- DELSYS MODE LISTER (FIXED) ---")
    
    # Initialize
    base = TrignoBase(DummyHandler())
    
    # Inject License (Just in case)
    import AeroPy.TrignoBase as TB
    TB.key = "YOUR_API_KEY_HERE"
    TB.license = """YOUR_LICENSE_XML_HERE"""

    print("[1/3] Connecting...")
    base.Connect_Callback()
    
    print("[2/3] Scanning...")
    base.Scan_Callback()
    
    print("\n" + "="*60)
    print("AVAILABLE MODES")
    print("="*60)
    
    # TrignoBase keeps a count of sensors found
    sensor_count = base.SensorCount
    
    if sensor_count == 0:
        print("[Error] No sensors found. Check power/magnet.")
    
    for i in range(sensor_count):
        # Get Sensor Name
        # We can't easily get the name via TrignoBase methods, but we can get the modes
        print(f"\nSENSOR INDEX {i}:")
        print("-" * 20)
        
        try:
            # This calls the official helper method in TrignoBase.py
            modes = base.getSampleModes(i)
            
            for mode_name in modes:
                print(f"  • {mode_name}")
                
        except Exception as e:
            print(f"  [Error retrieving modes: {e}]")

    print("\n" + "="*60)

if __name__ == "__main__":
    list_modes()