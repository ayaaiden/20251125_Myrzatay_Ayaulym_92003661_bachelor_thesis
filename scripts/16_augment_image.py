import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
import os
from collections import Counter

def augment_keypoints_aggressive(keypoints, num_augmentations=9):
    """Create 10x more data through augmentation"""
    augmented = [keypoints]
    
    for _ in range(num_augmentations):
        aug = keypoints.copy()
        
        # 1. Random noise (hand tremor simulation)
        noise_level = np.random.uniform(0.005, 0.02)
        noise = np.random.normal(0, noise_level, aug.shape)
        aug += noise
        
        # 2. Scale variation (distance from camera)
        scale = np.random.uniform(0.9, 1.1)
        aug *= scale
        
        # 3. Rotation (hand angle)
        angle = np.random.uniform(-15, 15) * np.pi / 180
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        aug_reshaped = aug.reshape(-1, 3)
        for i in range(len(aug_reshaped)):
            x, y, z = aug_reshaped[i]
            aug_reshaped[i, 0] = x * cos_a - y * sin_a
            aug_reshaped[i, 1] = x * sin_a + y * cos_a
        aug = aug_reshaped.flatten()
        
        # 4. Translation (hand position shift)
        translation = np.random.uniform(-0.08, 0.08, 3)
        for i in range(0, len(aug), 3):
            aug[i:i+3] += translation
        
        # 5. Smoothing (motion blur)
        if np.random.random() > 0.5:
            sigma = np.random.uniform(0.3, 1.0)
            aug = gaussian_filter1d(aug, sigma=sigma)
        
        # 6. Finger jitter
        if np.random.random() > 0.7:
            aug_reshaped = aug.reshape(-1, 3)
            jitter_indices = np.random.choice(len(aug_reshaped), 
                                            size=int(len(aug_reshaped)*0.3), 
                                            replace=False)
            for idx in jitter_indices:
                jitter = np.random.normal(0, 0.015, 3)
                aug_reshaped[idx] += jitter
            aug = aug_reshaped.flatten()
        
        augmented.append(aug)
    
    return np.array(augmented)

def augment_dataset(input_path, output_path, augmentation_factor=9):
    """Main augmentation function"""
    print("="*60)
    print("AGGRESSIVE DATA AUGMENTATION")
    print("="*60)
    
    # Load data
    print(f"\nLoading data from: {input_path}")
    data = np.load(input_path, allow_pickle=True).item()
    X_original = data['keypoints']
    y_original = data['labels']
    
    # Extract hand-only if needed
    if X_original.shape[1] == 1662:
        print("Extracting hand-only features (63)...")
        X_original = X_original[:, 195:258]
    
    print(f" Original dataset: {len(X_original)} samples")
    
    class_counts = Counter(y_original)
    print(f"\nCurrent samples per class: {list(class_counts.values())[0]}")
    print(f"After augmentation: {list(class_counts.values())[0] * (augmentation_factor + 1)}")
    
    # Augment
    X_augmented = []
    y_augmented = []
    
    print(f"\nAugmenting...")
    for i, (x, y) in enumerate(zip(X_original, y_original)):
        if (i + 1) % 100 == 0 or (i + 1) == len(X_original):
            print(f"Progress: {i+1}/{len(X_original)}")
        
        augmented_samples = augment_keypoints_aggressive(x, num_augmentations=augmentation_factor)
        X_augmented.extend(augmented_samples)
        y_augmented.extend([y] * len(augmented_samples))
    
    X_augmented = np.array(X_augmented)
    y_augmented = np.array(y_augmented)
    
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Original: {len(X_original)} samples")
    print(f"Augmented: {len(X_augmented)} samples")
    print(f"Increase: {len(X_augmented) - len(X_original)} samples ({(len(X_augmented)/len(X_original)):.1f}x)")
    
    # Save
    augmented_data = {
        'keypoints': X_augmented,
        'labels': y_augmented
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, augmented_data)
    print(f"\n Saved to: {output_path}")
    
    return X_augmented, y_augmented

if __name__ == "__main__":
    augment_dataset(
        input_path='data/keypoints/asl_alphabet_train_keypoints.npy',
        output_path='data/keypoints/asl_alphabet_train_keypoints_augmented_10x.npy',
        augmentation_factor=9 
    )
    
    print("\n" + "="*60)
    print("NEXT STEP:")
    print("="*60)
