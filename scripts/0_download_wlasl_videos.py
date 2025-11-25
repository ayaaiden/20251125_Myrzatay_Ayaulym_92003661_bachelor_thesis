#!/usr/bin/env python3
"""
0_download_wlasl_videos.py 
"""

import os
import json
import urllib.request
import ssl
import time
from tqdm import tqdm

print("="*60)
print("WLASL VIDEO DOWNLOADER SCRIPT")
print("="*60)

METADATA_PATH = "data/WLASL_v0.3.json"

if not os.path.exists(METADATA_PATH):
    print(f"✗ Error: Metadata file not found at {METADATA_PATH}")
    exit(1)

try:
    with open(METADATA_PATH, 'r') as f:
        wlasl_data = json.load(f)
    print(f"\n✓ Metadata loaded: {len(wlasl_data)} words available")
except Exception as e:
    print(f"✗ Error loading metadata: {e}")
    exit(1)

WORDS_TO_DOWNLOAD = [
    'all', 'black', 'blue', 'book', 'can', 'cool', 'dog', 'drink', 
    'go', 'help', 'hot', 'like', 'many', 'mother', 'no', 'now', 
    'orange', 'what', 'who', 'yes', 'hello', 'thank', 'you', 
    'please', 'good', 'water', 'food', 'ok', 'love'
]

VIDEOS_PER_WORD = 15
RAW_VIDEOS_DIR = "data/raw_videos"
os.makedirs(RAW_VIDEOS_DIR, exist_ok=True)

print(f" Target: {len(WORDS_TO_DOWNLOAD)} words")
print(f" Videos per word: {VIDEOS_PER_WORD}")
print(f" Timeout: 90 seconds per video")
print(f" Retries: 3 attempts per video")
print(f" Output directory: {RAW_VIDEOS_DIR}")

ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

total_downloaded = 0
failed_downloads = 0
skipped_existing = 0
words_found = 0

print("\n[3/3] Downloading videos...\n")

for entry in tqdm(wlasl_data, desc="Processing WLASL words"):
    word = entry.get('gloss', '').strip().lower()
    
    if word not in [w.lower().strip() for w in WORDS_TO_DOWNLOAD]:
        continue
    
    words_found += 1
    print(f"\n✓ Word #{words_found}/{len(WORDS_TO_DOWNLOAD)}: '{word}'")
    
    word_dir = os.path.join(RAW_VIDEOS_DIR, word)
    os.makedirs(word_dir, exist_ok=True)
    
    video_count = 0
    for idx, instance in enumerate(entry.get('instances', [])):
        if video_count >= VIDEOS_PER_WORD:
            break
        
        video_url = instance.get('url', '')
        if not video_url:
            continue
        
        video_name = f"{word}_{idx:03d}.mp4"
        video_path = os.path.join(word_dir, video_name)
        
        if os.path.exists(video_path):
            video_count += 1
            skipped_existing += 1
            continue
        
        # Download with retries
        max_retries = 3
        retry_count = 0
        success = False
        
        while retry_count < max_retries and not success:
            try:
                print(f"   {video_name}...", end=' ', flush=True)
                req = urllib.request.Request(video_url, headers={'User-Agent': 'Mozilla/5.0'})
                
                # Download video
                with urllib.request.urlopen(req, context=ssl_context, timeout=90) as response:
                    video_data = response.read()
                    
                    if len(video_data) > 1000:  # At least 1KB
                        with open(video_path, 'wb') as out_file:
                            out_file.write(video_data)
                        print(f" ({len(video_data) // 1024} KB)")
                        total_downloaded += 1
                        video_count += 1
                        success = True
                    else:
                        print(" (empty/corrupted)")
                        failed_downloads += 1
                        success = True
            
            except urllib.error.URLError as e:
                retry_count += 1
                error_type = type(e.reason).__name__ if hasattr(e, 'reason') else str(e)[:30]
                
                if "timeout" in str(e).lower() and retry_count < max_retries:
                    wait_time = 5 * retry_count
                    print(f" timeout (retry {retry_count}/{max_retries} in {wait_time}s)", end=' ')
                    time.sleep(wait_time)
                    print("\n retrying...")
                else:
                    print(f"✗ {error_type}")
                    failed_downloads += 1
                    success = True
            
            except Exception as e:
                print(f" {type(e).__name__}")
                failed_downloads += 1
                success = True
    
    print(f"   Completed '{word}': {video_count}/{VIDEOS_PER_WORD}")

print("\n" + "="*60)
print(" DOWNLOAD COMPLETE")
print("="*60)
print(f"\n Summary:")
print(f"    Words found: {words_found}/{len(WORDS_TO_DOWNLOAD)}")
print(f"    Videos downloaded: {total_downloaded}")
print(f"    Skipped (already exist): {skipped_existing}")
print(f"    Failed downloads: {failed_downloads}")
print(f"    Output directory: {RAW_VIDEOS_DIR}")

word_folders = sorted([d for d in os.listdir(RAW_VIDEOS_DIR) 
                      if os.path.isdir(os.path.join(RAW_VIDEOS_DIR, d))])

print(f"\n Dataset structure:")
for word in word_folders:
    word_path = os.path.join(RAW_VIDEOS_DIR, word)
    video_count = len([f for f in os.listdir(word_path) if f.endswith('.mp4')])
    print(f"   {word}: {video_count} videos")

print("="*60)

if total_downloaded < (len(WORDS_TO_DOWNLOAD) * VIDEOS_PER_WORD * 0.8):
    print(f"\n WARNING: Downloaded only {total_downloaded} videos out of ~{len(WORDS_TO_DOWNLOAD) * VIDEOS_PER_WORD} expected!")

