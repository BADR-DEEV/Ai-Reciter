import os
import requests
import time

# Create directory
output_folder = "Husary_Muallim_128kbps"
os.makedirs(output_folder, exist_ok=True)

base_url = "https://everyayah.com/data/Husary_Muallim_128kbps/"

# Number of Ayahs in each Surah (Index 0 = Surah 1, Index 113 = Surah 114)
ayah_counts = [
    7, 286, 200, 176, 120, 165, 206, 75, 129, 109, 123, 111, 43, 52, 99, 128, 111, 110, 98, 135,
    112, 78, 118, 64, 77, 227, 93, 88, 69, 60, 34, 30, 73, 54, 45, 83, 182, 88, 75, 85,
    54, 53, 89, 59, 37, 35, 38, 29, 18, 45, 60, 49, 62, 55, 78, 96, 29, 22, 24, 13,
    14, 11, 11, 18, 12, 12, 30, 52, 52, 44, 28, 28, 20, 56, 40, 31, 50, 40, 46, 42,
    29, 19, 36, 25, 22, 17, 19, 26, 30, 20, 15, 21, 11, 8, 8, 19, 5, 8, 8, 11,
    11, 8, 3, 9, 5, 4, 7, 3, 6, 3, 5, 4, 5, 6
]

print("Starting download...")

for surah_idx, total_ayahs in enumerate(ayah_counts):
    surah_num = surah_idx + 1
    
    for ayah_num in range(1, total_ayahs + 1):
        # Format filename: 001001.mp3 (3 digits for Surah, 3 digits for Ayah)
        file_name = f"{surah_num:03d}{ayah_num:03d}.mp3"
        url = base_url + file_name
        file_path = os.path.join(output_folder, file_name)

        # Skip if already exists
        if os.path.exists(file_path):
            continue

        try:
            print(f"Downloading {file_name}...")
            response = requests.get(url)
            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    f.write(response.content)
            else:
                print(f"Error downloading {file_name}: Status {response.status_code}")
        except Exception as e:
            print(f"Failed {file_name}: {e}")

print("Download Complete.")