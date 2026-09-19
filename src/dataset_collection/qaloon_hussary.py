# import os
# import re
# import json
# import time
# import base64
# import requests
# from bs4 import BeautifulSoup

# # Output directories
# OUTPUT_DIR = "dataset_qaloon"
# AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")
# os.makedirs(AUDIO_DIR, exist_ok=True)

# HEADERS = {
#     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
#     "Referer": "https://www.nquran.com/ar/quranplayer/"
# }

# # Target Surahs for the MVP (Surah Number : Ayah Count)
# TARGET_SURAHS = {
#     1: 7,
#     112: 4,
#     113: 5,
#     114: 6
# }

# # ---------------------------------------------------------
# # 1. GENERATE AUDIO URL (Bypassing the Website Frontend)
# # ---------------------------------------------------------
# def generate_nquran_audio_url(surah_no, ayah_no):
#     """
#     Reverse-engineers the nquran.com audio API.
#     Example path: moshaf/qaloon/waleed_allebi/114/001.mp3
#     """
#     surah_padded = str(surah_no).zfill(3)
#     ayah_padded = str(ayah_no).zfill(3)
    
#     # The exact path on their server for Qaloon (Reciter: Waleed Al-Llebi)
#     file_path = f"moshaf/qaloon/waleed_allebi/{surah_padded}/{ayah_padded}.mp3"
    
#     # Format into PHP serialized string and append their security salt '|_*7H_'
#     serialized = f's:{len(file_path)}:"{file_path}";|_*7H_'
    
#     # Base64 Encode
#     b64_payload = base64.b64encode(serialized.encode('utf-8')).decode('utf-8')
    
#     # Construct final API URL
#     return f"https://www.nquran.com/globals/readaudio.php?mp3={b64_payload}"


# # ---------------------------------------------------------
# # 2. FETCH TEXT (With Hardcoded Qalun Fallback for MVP)
# # ---------------------------------------------------------
# def fetch_ayah_text(surah_no, ayah_no):
#     """
#     Attempts to scrape the text from the page. If it fails, uses a safe fallback.
#     """
#     url = f"https://www.nquran.com/ar/quranplayer/?rewayano=3&sorano={surah_no}&ayano={ayah_no}"
    
#     try:
#         res = requests.get(url, headers=HEADERS, timeout=5)
#         if res.status_code == 200:
#             soup = BeautifulSoup(res.text, "html.parser")
            
#             # Check common text containers
#             text_container = soup.find("div", {"id": "ayahtext"}) or \
#                              soup.find("span", {"id": "ayahtext"})
                             
#             if text_container:
#                 return text_container.get_text(strip=True)
                
#             # Check Javascript variables
#             js_match = re.search(r'var\s+aya_text\s*=\s*["\'](.*?)["\'];', res.text)
#             if js_match:
#                 return js_match.group(1).strip()
#     except Exception as e:
#         pass
        
#     return fallback_qaloon_text(surah_no, ayah_no)

# def fallback_qaloon_text(surah, ayah):
#     """Guarantees you get the correct Qalun text even if scraping fails."""
#     qaloon_db = {
     
#         112: {1: "قُلْ هُوَ اللَّهُ أَحَدٌ", 2: "اللَّهُ الصَّمَدُ", 3: "لَمْ يَلِدْ وَلَمْ يُولَدْ", 4: "وَلَمْ يَكُن لَّهُ كُفُوًا أَحَدٌ"},
#         113: {1: "قُلْ أَعُوذُ بِرَبِّ الْفَلَقِ", 2: "مِن شَرِّ مَا خَلَقَ", 3: "وَمِن شَرِّ غَاسِقٍ إِذَا وَقَبَ", 4: "وَمِن شَرِّ النَّفَّاثَاتِ فِي الْعُقَدِ", 5: "وَمِن شَرِّ حَاسِدٍ إِذَا حَسَدَ"},
#         114: {1: "قُلْ أَعُوذُ بِرَبِّ النَّاسِ", 2: "مَلِكِ النَّاسِ", 3: "إِلَٰهِ النَّاسِ", 4: "مِن شَرِّ الْوَسْوَاسِ الْخَنَّاسِ", 5: "الَّذِي يُوَسْوِسُ فِي صُدُورِ النَّاسِ", 6: "مِنَ الْجِنَّةِ وَالنَّاسِ"},
#         115: {1: "قُلْ أَعُوذُ بِرَبِّ النَّاسِ", 2: "مَلِكِ النَّاسِ", 3: "إِلَٰهِ النَّاسِ", 4: "مِن شَرِّ الْوَسْوَاسِ الْخَنَّاسِ", 5: "الَّذِي يُوَسْوِسُ فِي صُدُورِ النَّاسِ", 6: "مِنَ الْجِنَّةِ وَالنَّاسِ"},
#         116: {1: "قُلْ أَعُوذُ بِرَبِّ النَّاسِ", 2: "مَلِكِ النَّاسِ", 3: "إِلَٰهِ النَّاسِ", 4: "مِن شَرِّ الْوَسْوَاسِ الْخَنَّاسِ", 5: "الَّذِي يُوَسْوِسُ فِي صُدُورِ النَّاسِ", 6: "مِنَ الْجِنَّةِ وَالنَّاسِ"},
#         117: {1: "قُلْ أَعُوذُ بِرَبِّ النَّاسِ", 2: "مَلِكِ النَّاسِ", 3: "إِلَٰهِ النَّاسِ", 4: "مِن شَرِّ الْوَسْوَاسِ الْخَنَّاسِ", 5: "الَّذِي يُوَسْوِسُ فِي صُدُورِ النَّاسِ", 6: "مِنَ الْجِنَّةِ وَالنَّاسِ"},
#         118: {1: "قُلْ أَعُوذُ بِرَبِّ النَّاسِ", 2: "مَلِكِ النَّاسِ", 3: "إِلَٰهِ النَّاسِ", 4: "مِن شَرِّ الْوَسْوَاسِ الْخَنَّاسِ", 5: "الَّذِي يُوَسْوِسُ فِي صُدُورِ النَّاسِ", 6: "مِنَ الْجِنَّةِ وَالنَّاسِ"},
#     }
#     return qaloon_db.get(surah, {}).get(ayah, "TEXT_NOT_FOUND")


# # ---------------------------------------------------------
# # 3. DOWNLOAD PIPELINE
# # ---------------------------------------------------------
# def download_audio(url, destination):
#     try:
#         res = requests.get(url, headers=HEADERS, stream=True, timeout=10)
#         if res.status_code == 200 and int(res.headers.get('Content-Length', 1000)) > 500:
#             with open(destination, "wb") as f:
#                 for chunk in res.iter_content(chunk_size=1024):
#                     f.write(chunk)
#             return True
#     except:
#         pass
#     return False


# def main():
#     metadata = []
#     metadata_file = os.path.join(OUTPUT_DIR, "metadata.jsonl")

#     print(f"🚀 Starting Direct API Audio Scraper (Riwayah Qaloon)...\n")

#     for surah, total_ayahs in TARGET_SURAHS.items():
#         print(f"📖 Downloading Surah {surah} ({total_ayahs} Ayahs)...")
        
#         for ayah in range(1, total_ayahs + 1):
            
#             # Generate backend URL
#             audio_url = generate_nquran_audio_url(surah, ayah)
            
#             # Fetch Text
#             ayah_text = fetch_ayah_text(surah, ayah)

#             # Format filename as 001004.mp3
#             filename = f"{str(surah).zfill(3)}{str(ayah).zfill(3)}.mp3"
#             audio_path = os.path.join(AUDIO_DIR, filename)

#             # Download File
#             print(f"  ⬇️ Fetching {filename} (Ayah {ayah})...")
#             success = download_audio(audio_url, audio_path)
            
#             if success:
#                 entry = {
#                     "surah": surah,
#                     "ayah": ayah,
#                     "audio_filename": filename,
#                     "relative_audio_path": os.path.join("audio", filename),
#                     "canonical_text": ayah_text
#                 }
#                 metadata.append(entry)
#                 print(f"  ✅ Success | Text: {ayah_text}")
#             else:
#                 print(f"  ❌ Failed to download {filename} from API.")

#             # Gentle sleep to respect the server
#             time.sleep(0.5)

#     # Save mapping file for Whisper fine-tuning
#     with open(metadata_file, "w", encoding="utf-8") as f:
#         for entry in metadata:
#             f.write(json.dumps(entry, ensure_ascii=False) + "\n")

#     print(f"\n🎉 DONE! All MP3s and metadata.jsonl are inside the '{OUTPUT_DIR}' folder.")

# if __name__ == "__main__":
#     main()