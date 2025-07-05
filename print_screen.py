import pyautogui
import time
import datetime
import os

output_dir = "dataset_iracing/screenshots"
os.makedirs(output_dir, exist_ok=True)

try:
    while True:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(output_dir, f"screenshot_{timestamp}.jpg")
        screenshot = pyautogui.screenshot()
        screenshot.save(filename)
        print(f"Saved {filename}")
        time.sleep(1)
except KeyboardInterrupt:
    print("Stopped by user.")
    
    