import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
import os
import pickle

# ======== Configuration ========
KEYPOINTS_FILE = "data/keypoints/wlasl-video_keypoints.npy/asl_alphabet_train_keypoints.npy"
LABELS_FILE = "data/keypoints/wlasl-video_keypoints.npy/asl_alphabet_train_labels.npy"
LABEL_ENCODER_FILE = "data/keypoints/wlasl-video_keypoints.npy/asl_alphabet_label_encoder.pkl"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ======== 1. Load preprocessed data ========
print("\nLoading preprocessed data...")
X = np.load(KEYPOINTS_FILE)
y = np.load(LABELS_FILE)

print(f" Loaded {len(X)} samples")
print(f" Keypoints shape: {X.shape}")
print(f" Labels shape: {y.shape}")

# ======== 2. Load or create label encoder ========
if os.path.exists(LABEL_ENCODER_FILE):
    with open(LABEL_ENCODER_FILE, 'rb') as f:
        class_names = pickle.load(f)
    print(f" Loaded {len(class_names)} classes: {class_names}")
else:
    print(" Label encoder file not found!")
    exit(1)

# ======== 3. Encode labels ========
le = LabelEncoder()
le.fit(class_names)
y_encoded = to_categorical(y, num_classes=len(class_names))

print(f" Encoded labels shape: {y_encoded.shape}")
print(f" Number of classes: {len(class_names)}")

# ======== 4. Train-test split  ========
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, 
    test_size=0.2, 
    random_state=42,
    stratify=y  
)

print(f"\n✓ Training set: {len(X_train)} samples")
print(f"✓ Validation set: {len(X_test)} samples")

# ======== 5. Reshape for LSTM (samples, timesteps=1, features) ========
X_train_lstm = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test_lstm = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

print(f" LSTM input shape: {X_train_lstm.shape}")

# ======== 6. Define model ========
model = Sequential([
    Masking(mask_value=0., input_shape=(1, X_train.shape[1])),
    LSTM(128, return_sequences=True),
    Dropout(0.3),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(len(class_names), activation='softmax')
])

 #Compile model 

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


print("\n" + "="*60)
print("MODEL ARCHITECTURE")
print("="*60)
model.summary()

# ======== 7. Train model ========
print("\n" + "="*60)
print("TRAINING MODEL")
print("="*60)

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
]

print("\nTraining...")

history = model.fit(
    X_train_lstm, y_train,  
    validation_data=(X_test_lstm, y_test),
    epochs=50,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

# ======== 8. Evaluate model ========
print("\n" + "="*60)
print("FINAL EVALUATION")
print("="*60)

test_loss, test_acc = model.evaluate(X_test_lstm, y_test, verbose=0)
print(f"\n Test Loss: {test_loss:.4f}")
print(f" Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

# ======== 9. Per-class accuracy ========
print("\n" + "="*60)
print("PER-CLASS ACCURACY")
print("="*60)

y_pred = model.predict(X_test_lstm, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

print("\n Per-class accuracy:")
for i, class_name in enumerate(class_names):
    class_mask = (y_test_classes == i)
    if np.sum(class_mask) > 0:
        class_acc = np.sum((y_pred_classes[class_mask] == i)) / np.sum(class_mask)
        status = "Yes" if class_acc > 0.5 else "No"
        print(f"{status} {class_name}: {class_acc*100:.1f}%")

# ======== 10. Save trained model ========
print("\n" + "="*60)
print("SAVING MODEL")
print("="*60)

model.save(os.path.join(MODEL_DIR, 'asl_alphabet_hand_only_final.keras'))
model.save(os.path.join(MODEL_DIR, 'asl_alphabet_hand_only_final.h5'))

with open(os.path.join(MODEL_DIR, 'asl_alphabet_hand_only_classes.pkl'), 'wb') as f:
    pickle.dump(class_names, f)

print("\n Model saved successfully!")
print(f"  - {MODEL_DIR}/asl_alphabet_hand_only_final.keras")
print(f"  - {MODEL_DIR}/asl_alphabet_hand_only_final.h5")
print(f"  - {MODEL_DIR}/asl_alphabet_hand_only_classes.pkl")

print("\n" + "="*60)
print("TRAINING COMPLETE!")
print("="*60)
