import numpy as np

print("="*60)
print("CHECKING DATA STRUCTURE")
print("="*60)

# Check Image/Photo Data
print("\n--- IMAGE/PHOTO DATA ---")
try:
    image_data = np.load("data/keypoints/asl_video_keypoints.npy", allow_pickle=True)
    print(f"Type: {type(image_data)}")
    
    if isinstance(image_data, np.ndarray):
        if image_data.dtype == object and image_data.shape == ():
            # Dict wrapped in array
            data_dict = image_data.item()
            print(f"Structure: Dictionary")
            print(f"Keys: {list(data_dict.keys())}")
            for key, value in data_dict.items():
                if isinstance(value, np.ndarray):
                    print(f"  - {key}: shape {value.shape}, dtype {value.dtype}")
                elif isinstance(value, list):
                    print(f"  - {key}: list with {len(value)} elements")
                else:
                    print(f"  - {key}: {type(value)}")
        else:
            print(f"Structure: Simple numpy array")
            print(f"Shape: {image_data.shape}")
            print(f"Dtype: {image_data.dtype}")
except FileNotFoundError:
    print("❌ File not found: data/keypoints/asl_alphabet_train_keypoints.npy")
except Exception as e:
    print(f"❌ Error: {e}")

# Check Video Data
print("\n--- VIDEO DATA ---")
try:
    video_data = np.load("data/keypoints/asl_video_keypoints.npy", allow_pickle=True)
    print(f"Type: {type(video_data)}")
    
    if isinstance(video_data, np.ndarray):
        if video_data.dtype == object and video_data.shape == ():
            # Dict wrapped in array
            data_dict = video_data.item()
            print(f"Structure: Dictionary")
            print(f"Keys: {list(data_dict.keys())}")
            for key, value in data_dict.items():
                if isinstance(value, np.ndarray):
                    print(f"  - {key}: shape {value.shape}, dtype {value.dtype}")
                elif isinstance(value, list):
                    print(f"  - {key}: list with {len(value)} elements")
                else:
                    print(f"  - {key}: {type(value)}")
        else:
            print(f"Structure: Simple numpy array")
            print(f"Shape: {video_data.shape}")
            print(f"Dtype: {video_data.dtype}")
except FileNotFoundError:
    print("File not found: data/keypoints/asl_video_keypoints.npy")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "="*60)
print("="*60)

