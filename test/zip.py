import os
from zipfile import ZipFile, ZIP_DEFLATED

# === Configuration ===
VIDEO_DIR = "/mnt/data/aschwab/data/video"     # Path to your video folder
OUTPUT_DIR = "/mnt/data/aschwab/data/zipped"    # Path to store zipped files

# === Ensure output folder exists ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Get all video files (adjust extensions if needed) ===
video_files = [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]

print(f"[INFO] Found {len(video_files)} video files to compress.")

# === Zip each video individually ===
for i, video in enumerate(video_files, 1):
    video_path = os.path.join(VIDEO_DIR, video)
    zip_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(video)[0]}.zip")

    with ZipFile(zip_path, 'w', compression=ZIP_DEFLATED) as zipf:
        zipf.write(video_path, arcname=video)

    if i % 5 == 0 or i == len(video_files):
        print(f"[INFO] Zipped {i}/{len(video_files)}: {video}")

print("[DONE] All videos zipped.")
