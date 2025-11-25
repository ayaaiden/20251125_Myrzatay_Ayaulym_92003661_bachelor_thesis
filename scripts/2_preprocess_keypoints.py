import os
import cv2
import numpy as np
import mediapipe as mp
from tqdm import tqdm
import pickle

# Suppress MediaPipe logging warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger('mediapipe').setLevel(logging.ERROR)

# Configuration
data_dir = os.path.expanduser("~/Downloads/asl_alphabet_train/asl_alphabet_train")
output_dir = "data/keypoints/wlasl-video_keypoints.npy"
os.makedirs(output_dir, exist_ok=True)

max_samples_per_class = 500  # Use 500 images per class

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.3, 
    min_tracking_confidence=0.3
)

def extract_keypoints(image_path):
    """Extract hand keypoints from image."""
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # Convert BGR to RGB (MediaPipe uses RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image with MediaPipe
        results = hands.process(image_rgb)
        
        if not results.multi_hand_landmarks:
            return None
        
        # Extract keypoints for all detected hands
        all_keypoints = []
        for hand_landmarks in results.multi_hand_landmarks:
            keypoints = []
            for landmark in hand_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.z])
            all_keypoints.extend(keypoints)
        
        
        while len(all_keypoints) < 126:
            all_keypoints.append(0.0)
        
        return np.array(all_keypoints[:126], dtype=np.float32)
    
    except Exception as e:
        return None

def main():
    print("\n" + "="*60)
    print("PREPROCESSING ASL ALPHABET IMAGES")
    print("="*60 + "\n")
    
    # Get all class folders
    class_folders = sorted([f for f in os.listdir(data_dir) 
                           if os.path.isdir(os.path.join(data_dir, f))])
    
    print(f"Found {len(class_folders)} classes")
    print(f"Processing up to {max_samples_per_class} images per class\n")
    
    all_keypoints = []
    all_labels = []
    
    # Process each class
    for class_idx, class_name in enumerate(class_folders):
        class_path = os.path.join(data_dir, class_name)
        
        # Get all images in class
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        # Limit to max_samples_per_class
        image_files = image_files[:max_samples_per_class]
        
        # Process images with progress bar
        successful = 0
        failed = 0
        
        for image_file in tqdm(image_files, 
                              desc=f"Processing {class_name:10s}", 
                              ncols=80,
                              leave=False):
            image_path = os.path.join(class_path, image_file)
            keypoints = extract_keypoints(image_path)
            
            if keypoints is not None:
                all_keypoints.append(keypoints)
                all_labels.append(class_idx)
                successful += 1
            else:
                failed += 1
        
        print(f"✓ {class_name:10s}: {successful:4d}/{len(image_files)} images processed")
    
    # Convert to numpy arrays
    X = np.array(all_keypoints, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32)
    
    # Save preprocessed data
    keypoints_file = os.path.join(output_dir, "asl_alphabet_train_keypoints.npy")
    labels_file = os.path.join(output_dir, "asl_alphabet_train_labels.npy")
    
    np.save(keypoints_file, X)
    np.save(labels_file, y)
    
    # Save label encoder
    label_encoder_file = os.path.join(output_dir, "asl_alphabet_label_encoder.pkl")
    with open(label_encoder_file, 'wb') as f:
        pickle.dump(class_folders, f)
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE!")
    print("="*60)
    print(f"\n✓ Saved {len(X)} samples to {keypoints_file}")
    print(f" Shape: {X.shape}")
    print(f" Classes: {len(class_folders)}\n")

if __name__ == "__main__":
    main()
