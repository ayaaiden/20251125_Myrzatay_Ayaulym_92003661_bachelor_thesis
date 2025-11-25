import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import json

# =======Configuration=======
JSON_PATH = "data/WLASL_v0.3.json"
RAW_VIDEOS_DIR = "data/WLASL/start_kit/raw_videos"
OUTPUT_FILE = "data/keypoints/asl_video_keypoints.npy"
MAX_FRAMES = 30
MAX_VIDEOS_PER_CLASS = 10


# ====== Mediapipe Setup ======
mp_holistic = mp.solutions.holistic

# ====== Load Metrics ======
print("Loading video metadata...")
with open(JSON_PATH, 'r') as f:
    wlasl_data = json.load(f)

# Build video ID -> gloss mapping
video2gloss = {}
for entry in wlasl_data:
    gloss = entry["gloss"]
    for instance in entry ["instances"]:
        vid =instance["video_id"]
        video2gloss[vid] = gloss
print(f"Total videos in metadata:{len(video2gloss)}")

# ====== Get Available Vidoes ======
available_videos = [f.replace(".mp4", "")for f in 
                    os.listdir(RAW_VIDEOS_DIR)if f.endswith(".mp4")]
print(f"Total available videos: {len(available_videos)}")

# Filter to videos we have labels for
valid_videos = [vid for vid in available_videos if vid in video2gloss]
print(f"Videos with valid labels: {len(valid_videos)}")

# ============ Limit per class (optional) ============
if MAX_VIDEOS_PER_CLASS:
    class_counts = {}
    filtered_videos = []
    for vid in valid_videos:
        gloss = video2gloss[vid]
        if class_counts.get(gloss, 0) < MAX_VIDEOS_PER_CLASS:
            filtered_videos.append(vid)
            class_counts[gloss] = class_counts.get(gloss, 0) + 1
    valid_videos = filtered_videos
    print(f"After limiting to {MAX_VIDEOS_PER_CLASS} per class: {len(valid_videos)} videos")

# ============ Extract Keypoints ============
all_sequences = []
all_labels = []

print("\nExtracting keypoints from videos...")
with mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.5) as holistic:
    for vid in tqdm(valid_videos):
        video_path = os.path.join(RAW_VIDEOS_DIR, f"{vid}.mp4")
        gloss = video2gloss[vid]
        
        # Read video
        cap = cv2.VideoCapture(video_path)
        frame_keypoints = []
        
        while cap.isOpened() and len(frame_keypoints) < MAX_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to RGB
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = holistic.process(img_rgb)
            
            # Extract pose, hands keypoints 
            # Left Hand 
            lh = np.array([[h.x, h.y, h.z] for h in results.left_hand_landmarks.landmark]) \
                 if results.left_hand_landmarks else np.zeros((21, 3))
            # Right Hand
            rh = np.array([[h.x, h.y, h.z] for h in results.right_hand_landmarks.landmark]) \
                 if results.right_hand_landmarks else np.zeros((21, 3))
            
            # Flatten and concatenate
            keypoints = np.concatenate([lh.flatten(), rh.flatten()])
            frame_keypoints.append(keypoints)
        
        cap.release()
        
        # Pad or truncate to MAX_FRAMES
        if len(frame_keypoints) == 0:
            # Skip videos with no valid frames
            continue
        
        while len(frame_keypoints) < MAX_FRAMES:
            # Pad with zeros
            frame_keypoints.append(np.zeros_like(frame_keypoints[0]))
        
        frame_keypoints = frame_keypoints[:MAX_FRAMES]
        
        # Store
        all_sequences.append(np.array(frame_keypoints))
        all_labels.append(gloss)

# ============ Convert to Arrays ============
all_sequences = np.array(all_sequences)
all_labels = np.array(all_labels)

print(f"\n Processed {len(all_sequences)} videos")
print(f" Sequence shape: {all_sequences.shape}")
print(f" Unique classes: {len(set(all_labels))}")

# ============ Save ============
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
np.save(OUTPUT_FILE, {
    'sequences': all_sequences,
    'labels': all_labels
})

print(f" Saved to: {OUTPUT_FILE}")

