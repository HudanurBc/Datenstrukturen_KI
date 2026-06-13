# download_sleepedf.py
# Helper script to automatically download a small, sufficient subset of the Sleep-EDF data (5 subjects = 10 recordings)
# Prevents a lengthy download of the full 8.1 GB ZIP archive.

import os
import urllib.request
import sys

# Base URL from PhysioNet
BASE_URL = "https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette/"

# Target directory in the Sleep-EDF project
DEST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "SLEEP_data", "physionet-sleep-data"))

# List of files to download (5 subjects, 2 nights each)
FILES_TO_DOWNLOAD = [
    # Subject 0
    "SC4001E0-PSG.edf", "SC4001EC-Hypnogram.edf",
    "SC4002E0-PSG.edf", "SC4002EC-Hypnogram.edf",
    # Subject 1
    "SC4011E0-PSG.edf", "SC4011EH-Hypnogram.edf",
    "SC4012E0-PSG.edf", "SC4012EC-Hypnogram.edf",
    # Subject 2
    "SC4021E0-PSG.edf", "SC4021EH-Hypnogram.edf",
    "SC4022E0-PSG.edf", "SC4022EC-Hypnogram.edf",
    # Subject 3
    "SC4031E0-PSG.edf", "SC4031EC-Hypnogram.edf",
    "SC4032E0-PSG.edf", "SC4032EC-Hypnogram.edf",
    # Subject 4
    "SC4041E0-PSG.edf", "SC4041EC-Hypnogram.edf",
    "SC4042E0-PSG.edf", "SC4042EC-Hypnogram.edf"
]

def progress_hook(count, block_size, total_size):
    """Displays the download progress of the current file."""
    percent = int(count * block_size * 100 / total_size)
    percent = min(100, percent)
    sys.stdout.write(f"\rDownloading: {percent}% of current file...")
    sys.stdout.flush()

def main():
    print("================ DOWNLOAD MANAGER FOR SLEEP-EDF ================")
    print(f"Target Directory: {DEST_DIR}")
    os.makedirs(DEST_DIR, exist_ok=True)
    
    total_files = len(FILES_TO_DOWNLOAD)
    
    for idx, filename in enumerate(FILES_TO_DOWNLOAD, 1):
        url = BASE_URL + filename
        dest_path = os.path.join(DEST_DIR, filename)
        
        print(f"\n[{idx}/{total_files}] Downloading: {filename}")
        
        if os.path.exists(dest_path):
            print(f"-> File already exists locally. Skipping...")
            continue
            
        try:
            urllib.request.urlretrieve(url, dest_path, progress_hook)
            print("\n-> Download completed!")
        except Exception as e:
            print(f"\nError downloading {filename}: {e}")
            print("Please ensure you have an active internet connection.")
            
    print("\n================ SLEEP-EDF DOWNLOADS FINISHED ================")
    print("All files are ready. You can now start the preprocessing!")

if __name__ == "__main__":
    main()
