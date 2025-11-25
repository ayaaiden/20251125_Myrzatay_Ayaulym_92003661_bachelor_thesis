"""
Real-time video-based sign language recognition
Integrated with trained WLASL word model + Alphabet fallback

"""
import cv2
import mediapipe as mp
import numpy as np
import os
from collections import deque, Counter
import time
import tensorflow as tf
import pickle

print("="*70)
print("REAL-TIME SIGN LANGUAGE RECOGNITION SYSTEM")
print("With Word + Alphabet Fallback Model")
print("="*70)

# CONFIDENCE THRESHOLD FOR FALLBACK
CONFIDENCE_THRESHOLD = 0.5

# Check if trained models exist
WORD_MODEL_PATH = 'models/wlasl_word_model.keras'
WORD_LABELS_PATH = 'models/wlasl_label_encoder.npy'
ALPHABET_MODEL_PATH = 'models/asl_alphabet_hand_only_final.keras'
ALPHABET_LABELS_PATH = 'models/asl_alphabet_hand_only_classes.pkl'

USE_TRAINED_MODEL = os.path.exists(WORD_MODEL_PATH) and os.path.exists(WORD_LABELS_PATH)

# Load Word Model
if USE_TRAINED_MODEL:
    print("\n Loading word recognition model...")
    word_model = tf.keras.models.load_model(WORD_MODEL_PATH)
    word_label_encoder = np.load(WORD_LABELS_PATH, allow_pickle=True)
    print(f" Word vocabulary: {len(word_label_encoder)} words")
    print(f" Words: {', '.join(sorted(word_label_encoder))}")
    
    # Sliding window for temporal smoothing
    SEQUENCE_LENGTH = 30
    keypoint_buffer = deque(maxlen=SEQUENCE_LENGTH)
    prediction_history = deque(maxlen=10)
else:
    print("\n No word model found. Running in landmark detection mode only.")
    SEQUENCE_LENGTH = 30
    keypoint_buffer = deque(maxlen=SEQUENCE_LENGTH)

# Load Alphabet Model (Fallback)
alphabet_model = None
alphabet_classes = None
USE_ALPHABET_FALLBACK = False

if os.path.exists(ALPHABET_MODEL_PATH) and os.path.exists(ALPHABET_LABELS_PATH):
    try:
        print("\n Loading alphabet fallback model...")
        alphabet_model = tf.keras.models.load_model(ALPHABET_MODEL_PATH)
        with open(ALPHABET_LABELS_PATH, 'rb') as f:
            alphabet_classes = pickle.load(f)
        
        # Get expected input shape
        alphabet_expected_shape = alphabet_model.input_shape
        print(f" Alphabet model expects input shape: {alphabet_expected_shape}")
        
        USE_ALPHABET_FALLBACK = True
        print(f" Alphabet vocabulary: {len(alphabet_classes)} letters")
        print(f" Fallback system enabled")
    except Exception as e:
        print(f" Could not load alphabet model: {e}")
        USE_ALPHABET_FALLBACK = False
else:
    print("\nNo alphabet model found. Fallback disabled.")

# Initialize MediaPipe
print("\n[1/3] Initializing MediaPipe Hand Detection...")
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False, # Video stream
    max_num_hands=2, # Max 2 hands
    min_detection_confidence=0.5, # Detection confidence
    min_tracking_confidence=0.5 # Tracking confidence
)
print("MediaPipe initialized")

def extract_keypoints(results):
    """Extract hand keypoints in the format expected by model"""
    keypoints = []
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks[:2]:  # Max 2 hands
            for landmark in hand_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.z])
    
    # Pad if less than 2 hands detected (126 features = 42 landmarks × 3 coords)
    while len(keypoints) < 126:
        keypoints.append(0.0)
    
    return np.array(keypoints[:126])

def extract_keypoints_single_hand(results):
    """Extract keypoints for single hand (for word model)"""
    keypoints = []
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks[:2]:
            for landmark in hand_landmarks.landmark:
                keypoints.extend([landmark.x, landmark.y, landmark.z])
    
    # Pad to 63 features (21 landmarks × 3 coords)
    while len(keypoints) < 63:
        keypoints.append(0.0)
    
    return np.array(keypoints[:63])

def get_hand_info(results):
    """Get hand detection info for display"""
    if not results.multi_hand_landmarks:
        return "No hands detected", 0
    
    num_hands = len(results.multi_hand_landmarks)
    hand_types = []
    
    if results.multi_handedness:
        for hand_info in results.multi_handedness:
            label = hand_info.classification[0].label
            hand_types.append(label)
    
    hand_str = " + ".join(hand_types) if hand_types else f"{num_hands} hand(s)"
    return hand_str, num_hands

# Open webcam
print("\n[2/3] Opening webcam...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print(" Error: Could not open webcam")
    exit()

print("✓ Webcam opened (1280x720)")

print("\n[3/3] Starting real-time recognition...")
print("="*70)
print("\n CONTROLS:")
print("   - Press 'q' to quit")
if USE_TRAINED_MODEL:
    print("   - Press 'r' to reset prediction buffer")
    print("   - Press 'SPACE' to hold current prediction")
print("\n💡 TIPS:")
print("   - Position hands clearly in frame")
print("   - Ensure good lighting")
if USE_TRAINED_MODEL:
    print("   - Perform signs slowly for 1-2 seconds")
    print("   - Wait for buffer to fill (30 frames)")
print("="*70)

# Variables for FPS and prediction holding
fps_time = time.time()
frame_count = 0
hold_prediction = False
held_word = None
held_confidence = 0
held_source = ""

# Statistics for thesis evaluation
word_predictions_count = 0
alphabet_fallback_count = 0

# Letter accumulation for fingerspelling
accumulated_letters = ""
last_letter = ""
last_letter_time = time.time()
confirmed_letters = ""  
temp_letter = ""  
LETTER_HOLD_TIME = 1.5
AUTO_WORD_COMPLETE_TIME = 3.0

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break
    
    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    
    # Convert BGR to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe
    results = hands.process(frame_rgb)
    
    # Get hand info
    hand_info, num_hands = get_hand_info(results)
    
    # Draw hand landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2)
            )
    
    # Initialize prediction variables
    predicted_word = "No model loaded"
    confidence = 0.0
    prediction_source = ""
    pred_color = (255, 255, 255)
    
    # ===== WORD + ALPHABET PREDICTION LOGIC =====
    if USE_TRAINED_MODEL and not hold_prediction:
        # Extract keypoints for word model
        keypoints_word = extract_keypoints_single_hand(results)
        keypoint_buffer.append(keypoints_word)
        
        if len(keypoint_buffer) == SEQUENCE_LENGTH:
            # Prepare sequence for word model
            sequence = np.array(list(keypoint_buffer))
            sequence = np.expand_dims(sequence, axis=0)  # Shape: (1, 30, 63)
            
            # Predict word
            word_probs = word_model.predict(sequence, verbose=0)[0]
            word_idx = np.argmax(word_probs)
            word_confidence = word_probs[word_idx]
            
            # ============================================
            # ALPHABET FALLBACK LOGIC 
            # ============================================
            if word_confidence < 0.60 and USE_ALPHABET_FALLBACK:
                # Low confidence on word → use alphabet fallback
                
                try:
                    # Extract keypoints in alphabet model format (2 hands, 126 features)
                    keypoints_alphabet = extract_keypoints(results)
                    
                    # Check alphabet model expected shape
                    expected_shape = alphabet_model.input_shape
                    
                    # Reshape based on expected input
                    if len(expected_shape) == 3:  # (batch, timesteps, features)
                        if expected_shape[1] == 1:  # (1, 1, 126)
                            alphabet_sequence = keypoints_alphabet.reshape(1, 1, 126)
                        elif expected_shape[1] is None:  # (1, None, 126)
                            alphabet_sequence = keypoints_alphabet.reshape(1, 1, 126)
                        else:
                            # If expects specific timesteps, replicate the frame
                            alphabet_sequence = np.tile(keypoints_alphabet, (expected_shape[1], 1))
                            alphabet_sequence = np.expand_dims(alphabet_sequence, axis=0)
                    elif len(expected_shape) == 2:  # (batch, features)
                        alphabet_sequence = keypoints_alphabet.reshape(1, 126)
                    else:
                        raise ValueError(f"Unexpected model input shape: {expected_shape}")
                    
                    # Predict letter
                    alphabet_probs = alphabet_model.predict(alphabet_sequence, verbose=0)[0]
                    letter_idx = np.argmax(alphabet_probs)
                    letter_confidence = alphabet_probs[letter_idx]
                    
                    # Get the predicted letter
                    predicted_letter = alphabet_classes[letter_idx]
                    temp_letter = predicted_letter 
                    
                    # Display letter
                    predicted_word = predicted_letter
                    confidence = letter_confidence
                    prediction_source = "Alphabet Fallback"
                    pred_color = (0, 165, 255)  # Orange
                    alphabet_fallback_count += 1
                    
                    # Letter accumulation logic
                    current_time = time.time()
                    if predicted_letter == last_letter:
                        if current_time - last_letter_time > LETTER_HOLD_TIME:
                            if predicted_letter not in accumulated_letters:
                                accumulated_letters += predicted_letter
                                last_letter_time = current_time
                    else:
                        last_letter = predicted_letter
                        last_letter_time = current_time
                    
                except Exception as e:
                    print(f" Alphabet prediction error: {e}")
                    print(f"   Expected shape: {alphabet_model.input_shape}")
                    print(f"   Attempted shape: {alphabet_sequence.shape if 'alphabet_sequence' in locals() else 'N/A'}")
                    
                    # Fallback to word prediction
                    predicted_word = word_label_encoder[word_idx]
                    confidence = word_confidence
                    prediction_source = "Word Model (Fallback Error)"
                    pred_color = (0, 0, 255)  # Red
                    word_predictions_count += 1
            else:
                # High confidence on word → use word prediction
                predicted_word = word_label_encoder[word_idx]
                confidence = word_confidence
                prediction_source = "Word Model"
                pred_color = (0, 255, 0)  # Green
                word_predictions_count += 1
                
                # Reset letter accumulation on word detection
                accumulated_letters = ""
    
    # Calculate FPS
    frame_count += 1
    if frame_count % 10 == 0:
        fps = 10 / (time.time() - fps_time)
        fps_time = time.time()
    else:
        fps = frame_count / (time.time() - fps_time) if frame_count > 1 else 0
    
    # ===== DRAW UI OVERLAY =====
    h, w, _ = frame.shape
    
    # Dark semi-transparent background for info panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 200), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
    
    # Title
    cv2.putText(frame, "Sign Language Recognition System (Word + Alphabet)", (10, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    # Hand detection info
    hand_color = (0, 255, 0) if num_hands > 0 else (0, 0, 255)
    cv2.putText(frame, f"Hands: {hand_info}", (10, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, hand_color, 2)
    
    # Prediction display
    if USE_TRAINED_MODEL:
        # Prediction text
        cv2.putText(frame, f"Prediction: {predicted_word}", (10, 105), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, pred_color, 2)
        
        # Source label
        cv2.putText(frame, f"Source: [{prediction_source}]", (10, 135), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, pred_color, 2)
        
        # Show accumulated letters if fingerspelling
        if accumulated_letters and "Alphabet" in prediction_source:
            cv2.putText(frame, f"Building: {accumulated_letters}", (10, 165), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
        if confirmed_letters:
            cv2.putText(frame, f"Spelled: {confirmed_letters}", (10, 195), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    # FPS counter (top right)
        # Confidence bar
        if confidence > 0:
            bar_width = int(300 * confidence)
            cv2.rectangle(frame, (10, 215), (310, 230), (50, 50, 50), -1)
            cv2.rectangle(frame, (10, 215), (10 + bar_width, 230), pred_color, -1)
            cv2.putText(frame, f"Confidence: {confidence:.1%}", (320, 227), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # FPS counter (top right)
    if fps > 0:
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 120, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # Hold indicator
    if USE_TRAINED_MODEL and hold_prediction:
        cv2.putText(frame, "[HELD]", (w - 120, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Display frame
    cv2.imshow('Sign Language Recognition', frame)
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        print("\n✓ Exiting...")
        break
    
    elif key == ord('r') and USE_TRAINED_MODEL:
        keypoint_buffer.clear()
        if 'prediction_history' in locals():
            prediction_history.clear()
        hold_prediction = False
        accumulated_letters = ""
        print(" Buffer reset")

    elif key == 13:  
        if temp_letter:
            confirmed_letters += temp_letter
            print(f" Confirmed letter: {temp_letter}")
            temp_letter = ""
    

# Cleanup
cap.release()
cv2.destroyAllWindows()
hands.close()

print("\n" + "="*70)
print("SESSION SUMMARY - BACHELOR THESIS PROTOTYPE")
print("="*70)
print(f" Total frames processed: {frame_count}")
if USE_TRAINED_MODEL:
    print(f" Word model vocabulary: {len(word_label_encoder)} words")
    print(f" Word predictions: {word_predictions_count}")
if USE_ALPHABET_FALLBACK:
    print(f" Alphabet fallback vocabulary: {len(alphabet_classes)} letters")
    print(f" Alphabet fallback triggered: {alphabet_fallback_count} times")
print(f" Total predictions made: {word_predictions_count + alphabet_fallback_count}")
if (word_predictions_count + alphabet_fallback_count) > 0:
    word_percentage = (word_predictions_count / (word_predictions_count + alphabet_fallback_count)) * 100
    fallback_percentage = (alphabet_fallback_count / (word_predictions_count + alphabet_fallback_count)) * 100
    print(f" Word model usage: {word_percentage:.1f}%")
    print(f" Alphabet fallback usage: {fallback_percentage:.1f}%")
print(" Session ended successfully")
print("="*70)
