"""
16_train_wlasl_word_model.py 
Train LSTM model for WLASL word recognition 
"""

import numpy as np
import os
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt

print("="*70)
print("WLASL WORD MODEL TRAINING")
print("="*70)

# ============================================================
# STEP 1: Load Preprocessed Data
# ============================================================
print("\n[1/7] Loading preprocessed keypoints...")

KEYPOINTS_DIR = "data/keypoints_wlasl_augmented"
X_list = []
y_list = []
if not os.path.exists(KEYPOINTS_DIR):
    print(f" Error: Directory not found: {KEYPOINTS_DIR}")
    exit(1)

for word_folder in os.listdir(KEYPOINTS_DIR):
    word_path = os.path.join(KEYPOINTS_DIR, word_folder)
    
    if not os.path.isdir(word_path):
        continue
    
    for keypoint_file in os.listdir(word_path):
        if not keypoint_file.endswith('.npy'):
            continue
        
        keypoint_path = os.path.join(word_path, keypoint_file)
        
        try:
            keypoints = np.load(keypoint_path)
            X_list.append(keypoints)
            y_list.append(word_folder)
        except Exception as e:
            print(f"Failed to load {keypoint_file}: {e}")

print(f"Loaded {len(X_list)} sequences from {len(set(y_list))} words")

if len(X_list) == 0:
    print("\n Error: No keypoint data found!")
    exit(1)

# ============================================================
# STEP 2: Filter by Minimum Samples
# ============================================================
print("\n[2/7] Filtering dataset for balanced training...")

word_counts = Counter(y_list)
print(f"\n Samples per word:")
for word, count in sorted(word_counts.items()):
    print(f"   {word}: {count} samples")

MIN_SAMPLES_PER_WORD = 10

valid_words = {word for word, count in word_counts.items() if count >= MIN_SAMPLES_PER_WORD}

print(f"\n Valid words (≥{MIN_SAMPLES_PER_WORD} samples): {len(valid_words)}")

X_filtered = []
y_filtered = []

for keypoints, word in zip(X_list, y_list):
    if word in valid_words:
        X_filtered.append(keypoints)
        y_filtered.append(word)

X = np.array(X_filtered, dtype=np.float32)
y = np.array(y_filtered)

print(f"\nFiltered dataset:")
print(f"  X shape: {X.shape}")
print(f"  Samples: {len(X)}")
print(f"  Classes: {len(valid_words)}")

if len(valid_words) < 2:
    print("\n Error: Not enough valid words for training!")
    exit(1)

# ============================================================
# STEP 3: Train-Test-Validation Split
# ============================================================
print("\n[3/7] Splitting data (train/val/test)...")

unique_words = sorted(np.unique(y))
word_to_idx = {word: idx for idx, word in enumerate(unique_words)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}

y_encoded = np.array([word_to_idx[word] for word in y])

can_stratify = all(count >= 3 for count in Counter(y).values())

if can_stratify:
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    print("Used stratified split")
else:
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y_encoded, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    print("Used random split")

print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

# ============================================================
# STEP 4: Build LSTM Model
# ============================================================
print("\n[4/7] Building LSTM model...")

n_timesteps = X_train.shape[1]  
n_features = X_train.shape[2]   
n_classes = len(unique_words)

# Build model
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(n_timesteps, n_features), name='lstm_layer_1'),
    Dropout(0.2, name='dropout_1'),
    
    LSTM(64, return_sequences=False, name='lstm_layer_2'),
    Dropout(0.2, name='dropout_2'),
    
    Dense(n_classes, activation='softmax', name='output_layer')
])

# Compile model 
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(" Model architecture:")
model.summary()

# ============================================================
# STEP 5: Callbacks Configuration
# ============================================================
print("\n[5/7] Configuring callbacks...")

os.makedirs('models', exist_ok=True)

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=25,  
        restore_best_weights=True,
        verbose=1
    ),
    
    ModelCheckpoint(
        'models/wlasl_word_model.keras',
        monitor='val_accuracy',
        verbose=1
    ),
    
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.7,  
        patience=8,  
        min_lr=5e-5,  
        verbose=1
    )
]

print("Callbacks:")



# ============================================================
# STEP 6: Train Model
# ============================================================
print("\n[6/7] Training model for 100 epochs...")
print("="*70)

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=16,
    callbacks=callbacks,
    verbose=1
)

print("\n" + "="*70)
print(" TRAINING COMPLETE")
print("="*70)

# ============================================================
# STEP 7: Evaluate & Save
# ============================================================
print("\n[7/7] Final evaluation...")

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

print(f"\nFinal Results:")
print(f"   Test Loss: {test_loss:.4f}")
print(f"   Test Accuracy: {test_accuracy*100:.2f}%")
print(f"   Epochs trained: {len(history.history['loss'])}")
print(f"   Best validation accuracy: {max(history.history['val_accuracy'])*100:.2f}%")

np.save('models/wlasl_label_encoder.npy', unique_words)

print(f"\nModel saved: models/wlasl_word_model.keras")
print(f"Labels saved: models/wlasl_label_encoder.npy")
print(f"Vocabulary: {len(unique_words)} words")

# ============================================================
# STEP 8: Plot Results
# ============================================================
print("\n[8/7] Generating plots...")

try:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.plot(history.history['loss'], label='Train Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax2.set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('models/training_history.png', dpi=300, bbox_inches='tight')
    print(" Plot saved: models/training_history.png")
except Exception as e:
    print(f"  Plot error: {e}")

print("\n" + "="*70)
print(". TRAINING COMPLETE")
print("="*70)
print("="*70)
