"""
Methodology: MediaPipe Hand Landmark Extraction → LSTM Neural Network

"""
import mediapipe as mp
import numpy as np
import os
from tqdm import tqdm

# All feature extraction is performed by MediaPipe
import cv2

print("="*70)
print("WLASL VIDEO PREPROCESSOR - THESIS METHODOLOGY")
print("MediaPipe Hand Landmark Extraction → LSTM Temporal Modeling")
print("="*70)

# ============================================================
# STAGE 1: Initialize MediaPipe Hands Solution
# ============================================================
print("\n[1/6] Initializing MediaPipe Hands...")
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,        # Video mode (temporal consistency)
    max_num_hands=2,                 # Detect up to 2 hands
    min_detection_confidence=0.5,   # Detection threshold
    min_tracking_confidence=0.5     # Tracking threshold
)

print(" MediaPipe Hands initialized")
print("  - Detection mode: VIDEO (temporal tracking enabled)")
print("  - Max hands: 2 (supports two-handed signs)")

# ============================================================
# STAGE 2: Directory Configuration
# ============================================================
print("\n[2/6] Setting up directories...")

RAW_VIDEOS_DIR = "data/raw_videos"          # Input: Downloaded WLASL videos
KEYPOINTS_OUTPUT_DIR = "data/keypoints_wlasl"  # Output: Extracted keypoints
os.makedirs(KEYPOINTS_OUTPUT_DIR, exist_ok=True)

print(f" Input directory: {RAW_VIDEOS_DIR}")
print(f" Output directory: {KEYPOINTS_OUTPUT_DIR}")

# Validation
if not os.path.exists(RAW_VIDEOS_DIR):
    print(f"\n Error: Directory not found: {RAW_VIDEOS_DIR}")
    exit(1)

# ============================================================
# STAGE 3: Data Augmentation Functions
# ============================================================
def augment_keypoints(keypoints, aug_type=0):

    #Apply data augmentation to MediaPipe hand keypoints
    
    if aug_type == 0:
        return keypoints.copy()  
    
    augmented = keypoints.copy()
    
    # Define augmentation parameters
    if aug_type == 1:
        noise_level = 0.015  # 1.5% coordinate jitter
        rotation_angle = 0
    elif aug_type == 2:
        noise_level = 0.025  # 2.5% coordinate jitter
        rotation_angle = 0
    elif aug_type == 3:
        noise_level = 0.015
        rotation_angle = 5     # +5 degree rotation
    elif aug_type == 4:
        noise_level = 0.025
        rotation_angle = -5    # -5 degree rotation
    else:
        return augmented
    
    # Add Gaussian noise to simulate natural signing variation
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, augmented.shape)
        augmented += noise
        augmented = np.clip(augmented, 0.0, 1.0) 
    
    # Apply 2D rotation 
    if rotation_angle != 0:
        angle_rad = np.radians(rotation_angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Rotate each frame independently (preserves temporal order)
        for frame_idx in range(augmented.shape[0]):
            frame_keypoints = augmented[frame_idx]
            
            # Rotate x,y coordinates (z remains unchanged)
            for i in range(0, len(frame_keypoints), 3):
                x = frame_keypoints[i]
                y = frame_keypoints[i + 1]
                
                # 2D rotation transformation
                frame_keypoints[i] = x * cos_a - y * sin_a      
                frame_keypoints[i + 1] = x * sin_a + y * cos_a  
            
            augmented[frame_idx] = frame_keypoints
        
        augmented = np.clip(augmented, 0.0, 1.0)
    
    return augmented

# ============================================================
# STAGE 4: MediaPipe Feature Extraction Function
# ============================================================

def extract_hand_keypoints_from_video(video_path, max_frames=30):
    
    #Extract hand keypoints from video using MediaPipe Hands
    cap = cv2.VideoCapture(video_path)
    keypoints_sequence = []
    
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sample frames evenly to get exactly max_frames
    if total_frames > max_frames:
        frame_indices = np.linspace(0, total_frames-1, max_frames, dtype=int)
    else:
        frame_indices = range(total_frames)
    
    # Process each sampled frame
    while cap.isOpened() and frame_count < total_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count in frame_indices:
            # Convert BGR (cv2 format) to RGB (MediaPipe requirement)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # ===========================================
            # CORE: MediaPipe Hand Detection
            # ===========================================
            results = hands.process(frame_rgb)
            
            if results.multi_hand_landmarks:
                # Extract first detected hand
                hand_landmarks = results.multi_hand_landmarks[0]
                
                # Flatten 21 landmarks × 3 coordinates = 63 features
                keypoints = []
                for landmark in hand_landmarks.landmark:
                    keypoints.extend([landmark.x, landmark.y, landmark.z])
                
                keypoints_sequence.append(keypoints)
            else:
                # No hand detected - use zero padding
                keypoints_sequence.append([0.0] * 63)
        
        frame_count += 1
    
    cap.release()
    
    # Ensure fixed-length sequence
    if len(keypoints_sequence) < max_frames:
        # Zero-pad shorter sequences
        padding = [[0.0] * 63] * (max_frames - len(keypoints_sequence))
        keypoints_sequence.extend(padding)
    else:
        # Truncate longer sequences
        keypoints_sequence = keypoints_sequence[:max_frames]
    
    return np.array(keypoints_sequence, dtype=np.float32)

# ============================================================
# STAGE 5: Scan Video Dataset
# ============================================================
print("\n[3/6] Scanning WLASL video dataset...")

word_folders = [f for f in os.listdir(RAW_VIDEOS_DIR) 
                if os.path.isdir(os.path.join(RAW_VIDEOS_DIR, f))]

print(f" Found {len(word_folders)} word categories")

total_videos = 0
for word_folder in word_folders:
    word_path = os.path.join(RAW_VIDEOS_DIR, word_folder)
    video_count = len([f for f in os.listdir(word_path) if f.endswith('.mp4')])
    total_videos += video_count
    print(f"  {word_folder}: {video_count} videos")

print(f"\n Total videos to process: {total_videos}")
print(f"✓ Expected output: {total_videos * 5} keypoint files")
print(f"\n⏱  Estimated processing time: {int(total_videos * 0.5)} seconds")

# ============================================================
# STAGE 6: Process All Videos with Augmentation
# ============================================================
print("\n[4/6] Extracting hand keypoints with augmentation...")

processed_count = 0
failed_count = 0

for word_folder in tqdm(word_folders, desc="Processing word categories"):
    word_input_path = os.path.join(RAW_VIDEOS_DIR, word_folder)
    word_output_path = os.path.join(KEYPOINTS_OUTPUT_DIR, word_folder)
    os.makedirs(word_output_path, exist_ok=True)
    
    # Process each video in this word category
    for video_file in os.listdir(word_input_path):
        if not video_file.endswith('.mp4'):
            continue
        
        video_path = os.path.join(word_input_path, video_file)
        base_name = video_file.replace('.mp4', '')
        
        try:
            # ===================================================
            # STEP 1: MediaPipe Feature Extraction
            # ===================================================
            original_keypoints = extract_hand_keypoints_from_video(video_path, max_frames=30)
            
            # ===================================================
            # STEP 2: Data Augmentation (5× Expansion)
            # ===================================================
            for aug_idx in range(5):  # 0=original, 1-4=augmented
                if aug_idx == 0:
                    keypoint_filename = f"{base_name}.npy"
                    augmented_keypoints = original_keypoints
                else:
                    keypoint_filename = f"{base_name}_aug{aug_idx}.npy"
                    augmented_keypoints = augment_keypoints(original_keypoints, aug_type=aug_idx)
                
                keypoint_path = os.path.join(word_output_path, keypoint_filename)
                
                # Skip if already processed
                if os.path.exists(keypoint_path):
                    processed_count += 1
                    continue
                
                # Save keypoints as NumPy array
                np.save(keypoint_path, augmented_keypoints)
                processed_count += 1
        
        except Exception as e:
            failed_count += 1

print("\n" + "="*70)
print(" PREPROCESSING COMPLETE ")
print("="*70)

# ============================================================
# STAGE 7: Validation and Summary
# ============================================================
print("\n[5/6] Validation and summary statistics...\n")

print(f"📊 Processing Results:")
print(f"    Successfully processed: {processed_count} keypoint files")
print(f"    Failed: {failed_count} videos")
print(f"    Success rate: {(processed_count / (processed_count + failed_count) * 100):.1f}%")

# Count final keypoints per word
print(f"\n Final dataset structure:")
total_keypoints = 0
for word_folder in sorted(os.listdir(KEYPOINTS_OUTPUT_DIR)):
    word_path = os.path.join(KEYPOINTS_OUTPUT_DIR, word_folder)
    if os.path.isdir(word_path):
        keypoint_count = len([f for f in os.listdir(word_path) if f.endswith('.npy')])
        total_keypoints += keypoint_count
        print(f"   {word_folder}: {keypoint_count} samples ({keypoint_count // 5} orig + {keypoint_count - keypoint_count // 5} aug)")

print(f"\n Total dataset size: {total_keypoints} temporal sequences")

# Sample keypoint inspection
print(f"\n[6/6] Sample data validation...")
sample_word = word_folders[0] if word_folders else None
if sample_word:
    sample_word_path = os.path.join(KEYPOINTS_OUTPUT_DIR, sample_word)
    sample_files = [f for f in os.listdir(sample_word_path) if f.endswith('.npy')]
    
    if sample_files:
        sample_keypoint_path = os.path.join(sample_word_path, sample_files[0])
        sample_keypoints = np.load(sample_keypoint_path)
        
        print(f" Sample file: {sample_files[0]}")
        print(f" Shape: {sample_keypoints.shape}")
        print(f"  - Temporal dimension (frames): {sample_keypoints.shape[0]}")
        print(f"  - Spatial dimension (features): {sample_keypoints.shape[1]}")
        print(f"  - Data type: {sample_keypoints.dtype}")
        print(f"  - Value range: [{sample_keypoints.min():.3f}, {sample_keypoints.max():.3f}]")

print(f"\n Output directory: {KEYPOINTS_OUTPUT_DIR}/")
print("\n" + "="*70)
print("="*70)
print("\n" + "="*70)
print("  PREPROCESSING PHASE COMPLETE")
print("="*70)

# Clean up MediaPipe resources
hands.close()