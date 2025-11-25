"""
17_augment_wlasl_video.py
"""

import numpy as np
import os
import random

print("="*70)
print("WLASL VIDEO SEQUENCE AUGMENTATION")
print("Multiplying dataset 16x (1 original + 15 augmented)")
print("="*70)

KEYPOINTS_DIR = "data/keypoints_wlasl"
OUTPUT_DIR = "data/keypoints_wlasl_augmented"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# Temporal Augmentation Functions
# ============================================================

def temporal_warp(keypoints, factor):
    n_frames = keypoints.shape[0]
    new_n_frames = int(n_frames * factor)
    
    indices = np.linspace(0, n_frames - 1, new_n_frames)
    warped = np.zeros((n_frames, keypoints.shape[1]))
    
    for i, idx in enumerate(indices[:n_frames]):
        idx_low = int(idx)
        idx_high = min(idx_low + 1, n_frames - 1)
        alpha = idx - idx_low
        warped[i] = (1 - alpha) * keypoints[idx_low] + alpha * keypoints[idx_high]
    
    return warped

def spatial_jitter(keypoints, magnitude):
    """Add random noise to simulate hand tremor."""
    noise = np.random.normal(0, magnitude, keypoints.shape)
    return np.clip(keypoints + noise, 0, 1)

def hand_rotation(keypoints, degrees):
    """Rotate hand keypoints in 2D plane."""
    angle_rad = np.radians(np.random.uniform(-degrees, degrees))
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    
    rotated = keypoints.copy()
    for frame_idx in range(len(keypoints)):
        frame = keypoints[frame_idx].reshape(21, 3)
        xy = frame[:, :2]
        
        # Find center
        center = np.mean(xy, axis=0)
        
        # Rotate around center
        xy_centered = xy - center
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        xy_rotated = (rotation_matrix @ xy_centered.T).T + center
        
        # Clip to valid range
        xy_rotated = np.clip(xy_rotated, 0, 1)
        
        # Reconstruct
        frame[:, :2] = xy_rotated
        rotated[frame_idx] = frame.flatten()
    
    return rotated

def hand_scale(keypoints, scale):
    """Scale hand size """
    scaled = keypoints.copy()
    for frame_idx in range(len(keypoints)):
        frame = keypoints[frame_idx].reshape(21, 3)
        xy = frame[:, :2]
        
        center = np.mean(xy, axis=0)
        xy_scaled = (xy - center) * scale + center
        xy_scaled = np.clip(xy_scaled, 0, 1)
        
        frame[:, :2] = xy_scaled
        scaled[frame_idx] = frame.flatten()
    
    return scaled

def hand_shift(keypoints, max_shift):
    """Translate hand position."""
    shift_x = np.random.uniform(-max_shift, max_shift)
    shift_y = np.random.uniform(-max_shift, max_shift)
    
    shifted = keypoints.copy()
    for frame_idx in range(len(keypoints)):
        frame = keypoints[frame_idx].reshape(21, 3)
        frame[:, 0] = np.clip(frame[:, 0] + shift_x, 0, 1)
        frame[:, 1] = np.clip(frame[:, 1] + shift_y, 0, 1)
        shifted[frame_idx] = frame.flatten()
    
    return shifted

# ============================================================
# Apply Augmentation
# ============================================================

total_original = 0
total_augmented = 0

for word_folder in sorted(os.listdir(KEYPOINTS_DIR)):
    word_path = os.path.join(KEYPOINTS_DIR, word_folder)
    
    if not os.path.isdir(word_path):
        continue
    
    output_word_path = os.path.join(OUTPUT_DIR, word_folder)
    os.makedirs(output_word_path, exist_ok=True)
    
    keypoint_files = [f for f in os.listdir(word_path) if f.endswith('.npy')]
    
    for keypoint_file in keypoint_files:
        keypoint_path = os.path.join(word_path, keypoint_file)
        keypoints = np.load(keypoint_path)
        
        # Save original
        output_path = os.path.join(output_word_path, keypoint_file)
        np.save(output_path, keypoints)
        total_original += 1
        total_augmented += 1
        
        for aug_idx in range(15):
            augmented = keypoints.copy()
            
            
            n_augs = np.random.randint(2, 4)
            aug_funcs = [
                lambda x: temporal_warp(x, np.random.uniform(0.85, 1.15)),
                lambda x: spatial_jitter(x, np.random.uniform(0.01, 0.03)),
                lambda x: hand_rotation(x, np.random.uniform(10, 25)),
                lambda x: hand_scale(x, np.random.uniform(0.85, 1.15)),
                lambda x: hand_shift(x, np.random.uniform(0.05, 0.15))
            ]
            
            selected_funcs = random.sample(aug_funcs, min(n_augs, len(aug_funcs)))
            
            for aug_func in selected_funcs:
                try:
                    augmented = aug_func(augmented)
                except Exception as e:
                    print(f"     Aug error on {keypoint_file}: {e}")
                    continue
            
            # Save augmented
            base_name = keypoint_file.replace('.npy', f'_aug_{aug_idx:02d}.npy')
            output_path = os.path.join(output_word_path, base_name)
            np.save(output_path, augmented)
            total_augmented += 1
    
    print(f" {word_folder}: {len(keypoint_files)} → {len(keypoint_files)*16} sequences")

print(f"\n{'='*70}")
print(f"AUGMENTATION COMPLETE")
print(f"{'='*70}")
print(f"Original sequences: {total_original}")
print(f"Total after augmentation: {total_augmented}")
print(f"Multiplication factor: {total_augmented/total_original:.1f}x")
print(f"Output directory: {OUTPUT_DIR}")


