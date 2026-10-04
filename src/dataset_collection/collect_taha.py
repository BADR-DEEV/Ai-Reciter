"""
Automated Downloader for Juz' Amma (Surahs 78-114) from Assabile.com
Reciter: Mohammad Omar El-Ganayni (Qaloon 'an Nafi')
"""

import os
import re
import time
import zipfile
from pathlib import Path
from playwright.sync_api import sync_playwright

URL = "https://ar.assabile.com/mohammad-omar-el-ganayni-490/collection/al-mus-haf-al-murattal-535"
OUTPUT_DIR = Path("mohammed_omar_el_ganayni_juz_amma")
ZIP_DIR = OUTPUT_DIR / "zips"
MP3_DIR = OUTPUT_DIR / "mp3"

# Juz' Amma: Surahs 78 to 114
START_SURAH = 78
END_SURAH = 114


def extract_zip(zip_path: Path, extract_to: Path):
    """Extract mp3 from downloaded zip and remove zip if desired."""
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.namelist():
                if member.endswith(".mp3"):
                    zf.extract(member, extract_to)
                    print(f"      Extracted MP3: {member}")
    except Exception as e:
        print(f"      ⚠️ Failed to unzip {zip_path.name}: {e}")


def main():
    ZIP_DIR.mkdir(parents=True, exist_ok=True)
    MP3_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("🚀 Starting Juz' Amma Downloader (Assabile.com)")
    print(f"Target: Surahs {START_SURAH} to {END_SURAH}")
    print(f"Destination: {OUTPUT_DIR.resolve()}")
    print("=" * 65)

    with sync_playwright() as p:
        # Launch browser (headless=False lets you see the browser and bypass Cloudflare if prompted)
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            accept_downloads=True
        )
        page = context.new_page()

        print(f"🌐 Navigating to collection page...")
        page.goto(URL, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(3000)

        # Scroll to bottom to ensure all 114 items are loaded in the DOM
        print("📜 Scrolling down to load all surahs...")
        for _ in range(5):
            page.mouse.wheel(0, 3000)
            page.wait_for_timeout(800)

        for surah_num in range(START_SURAH, END_SURAH + 1):
            target_zip = ZIP_DIR / f"surah_{surah_num:03d}.zip"
            
            # Check if MP3 already exists to avoid re-downloading
            existing_mp3s = list(MP3_DIR.glob(f"*{surah_num:03d}*.mp3")) or list(MP3_DIR.glob(f"*-{surah_num}-*.mp3"))
            if existing_mp3s or target_zip.exists():
                print(f"⏩ Surah {surah_num} already downloaded. Skipping.")
                continue

            print(f"\n📥 [{surah_num}/{END_SURAH}] Processing Surah #{surah_num}...")

            try:
                # 1. Locate the Surah row / item (Assabile uses format like "#78" or "#114")
                surah_pattern = re.compile(rf"#\s*{surah_num}\b")
                surah_element = page.locator(f"text={surah_pattern}").first

                if not surah_element.count():
                    # Fallback locator
                    surah_element = page.get_by_text(f"#{surah_num}").first

                if not surah_element.count():
                    print(f"   ⚠️ Could not locate row for Surah #{surah_num}")
                    continue

                # Scroll row into view
                surah_element.scroll_into_view_if_needed()
                page.wait_for_timeout(500)

                # 2. Find the download icon in this row / parent container
                # Assabile places action icons in the same list item / table row
                parent_row = surah_element.locator("xpath=ancestor::*[self::li or self::tr or contains(@class, 'recitation') or contains(@class, 'track')][1]")
                
                # Search for download button inside this row
                download_trigger = parent_row.locator("a[title*='تحميل'], a[title*='Download'], a[title*='Télécharger'], a:has(i[class*='download']), [class*='download']").first
                
                if not download_trigger.count():
                    # Wider search near the surah element
                    download_trigger = surah_element.locator("xpath=following::a[contains(@title, 'تحميل') or contains(@title, 'Download') or contains(@class, 'download')][1]")

                # Click the first download trigger to open the modal
                download_trigger.click(timeout=10000)
                page.wait_for_timeout(1500)

                # 3. Handle the modal popup and click the final "تحميل" / "Download" button
                # Assabile opens a popup with text "تحميل" or "Veuillez ne pas dépasser"
                modal_download_btn = page.locator(".modal a:has-text('تحميل'), .modal a:has-text('Télécharger'), .modal a:has-text('Download'), .popup a[href*='recitations'], a.btn-download").first

                if modal_download_btn.count():
                    with page.expect_download(timeout=60000) as download_info:
                        modal_download_btn.click()
                    download = download_info.value
                else:
                    # In case clicking the first button directly started the download
                    print("   Modal button not detected, checking direct download...")
                    with page.expect_download(timeout=15000) as download_info:
                        download_trigger.click()
                    download = download_info.value

                # Save the ZIP file
                download.save_as(str(target_zip))
                print(f"   ✅ Saved: {target_zip.name} ({os.path.getsize(target_zip) // 1024} KB)")

                # 4. Extract MP3 immediately
                extract_zip(target_zip, MP3_DIR)

                # Close any open modal/backdrop if present
                close_btn = page.locator(".modal .close, .modal [data-dismiss='modal'], .close").first
                if close_btn.count() and close_btn.is_visible():
                    close_btn.click()
                else:
                    page.keyboard.press("Escape")

                # Respect Assabile server rule: "Please do not exceed two simultaneous downloads"
                time.sleep(2.5)

            except Exception as e:
                print(f"   ❌ Error on Surah #{surah_num}: {e}")
                page.keyboard.press("Escape")
                time.sleep(2)

        browser.close()

    print("\n" + "=" * 65)
    print(f"🎉 Download Complete!")
    print(f"All MP3 files are ready in: {MP3_DIR.resolve()}")
    print("=" * 65)


if __name__ == "__main__":
    main()