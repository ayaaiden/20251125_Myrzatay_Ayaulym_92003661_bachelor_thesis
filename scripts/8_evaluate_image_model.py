import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score
import matplotlib.pyplot as plt
from tensorflow import keras
import os


print("="*60)
print("EVALUATING IMAGE/ALPHABET MODEL")
print("="*60)

os.makedirs('results', exist_ok=True)


# ============================================================
# STEP 1: Load Model First (to check expected input shape)
# ============================================================
print("\n[1/6] Loading model...")

model_path = 'models/asl_alphabet_hand_only_final.h5'  

if not os.path.exists(model_path):
    print(f"❌ ERROR: Model not found at {model_path}")
    print("\n💡 Please train the model first:")
    print("   python3 scripts/14_train_hand_only_model.py")
    exit(1)

model = keras.models.load_model(model_path)
print(f" Loaded model from: {model_path}")
print(f"Model input shape: {model.input_shape}")
print(f"Model output shape: {model.output_shape}")

expected_features = model.input_shape[1] if len(model.input_shape) == 2 else model.input_shape[2]
print(f"Expected features per sample: {expected_features}")


# ============================================================
# STEP 2: Load Test Data (matching model input)
# ============================================================
print("\n[2/6] Loading test data...")

# Load test data
X_test = np.load("data/keypoints/wlasl-video_keypoints.npy/asl_alphabet_train_keypoints.npy")
y_test_from_file = np.load("data/keypoints/wlasl-video_keypoints.npy/asl_alphabet_train_labels.npy")

print(f"✓ Loaded test data")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test_from_file.shape}")

# Verify shape compatibility
if X_test.shape[1] != expected_features:
    print(f"\n ERROR: Data shape mismatch!")
    print(f"   Model expects: {expected_features} features")
    print(f"   Data has: {X_test.shape[1]} features")
    exit(1)


# ============================================================
# STEP 3: Load/Create Label Encoder
# ============================================================
print("\n[3/6] Loading label encoder...")

# Define class names
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
               'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
               'del', 'nothing', 'space']

# Load label encoder
le_path = 'models/asl_alphabet_hand_only_classes.pkl'  

if os.path.exists(le_path):
    with open(le_path, 'rb') as f:
        class_names = pickle.load(f)
    print(f" Loaded label encoder from: {le_path}")
else:
    print(f" Label encoder not found. Creating from class names...")
    le = LabelEncoder()
    le.fit(class_names)
    with open(le_path, 'wb') as f:
        pickle.dump(class_names, f)
    print(f" Created and saved label encoder")

le = LabelEncoder()
le.fit(class_names)

print(f"Number of classes: {len(le.classes_)}")
print(f"Classes: {sorted(le.classes_)}")


# ============================================================
# STEP 4: Prepare Test Labels
# ============================================================
print("\n[4/6] Encoding labels...")

# Convert to string labels if needed
if isinstance(y_test_from_file[0], (int, np.integer)):
    print("Converting integer labels to class names...")
    y_test_str = [class_names[i] for i in y_test_from_file]
else:
    y_test_str = y_test_from_file

# Verify all labels are known
unknown_labels = set(y_test_str) - set(le.classes_)
if unknown_labels:
    print(f"\n❌ ERROR: Unknown labels in test data: {unknown_labels}")
    print(f"Expected classes: {sorted(le.classes_)}")
    exit(1)

# Encode labels
y_test_encoded = le.transform(y_test_str)
print(f"✓ Encoded {len(y_test_encoded)} labels")
print(f"Sample labels: {y_test_str[:5]} → {y_test_encoded[:5]}")


# ============================================================
# STEP 5: Reshape for model input (if needed)
# ============================================================
if len(model.input_shape) == 3: 
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    print(f"✓ Reshaped for LSTM: {X_test.shape}")


# ============================================================
# STEP 6: Make Predictions
# ============================================================
print("\n[5/6] Making predictions...")
y_pred = model.predict(X_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
print(f" Generated {len(y_pred_classes)} predictions")


# ============================================================
# STEP 7: Compute Metrics
# ============================================================
print("\n[6/6] Computing metrics...")

top1_accuracy = accuracy_score(y_test_encoded, y_pred_classes)
print(f"\n{'='*60}")
print(f"TOP-1 ACCURACY: {top1_accuracy:.4f} ({top1_accuracy*100:.2f}%)")
print(f"{'='*60}")

correct = np.sum(y_test_encoded == y_pred_classes)
total = len(y_test_encoded)
print(f"Correct predictions: {correct}/{total}")

f1_weighted = f1_score(y_test_encoded, y_pred_classes, average='weighted', zero_division=0)
print(f"Weighted F1-Score: {f1_weighted:.4f}")

# Per-class accuracy
print("\nPer-class accuracy:")
for cls in sorted(le.classes_):
    cls_idx = le.transform([cls])[0]
    idx = y_test_encoded == cls_idx
    if np.any(idx):
        cls_acc = (y_pred_classes[idx] == y_test_encoded[idx]).mean()
        count = np.sum(idx)
        print(f"  {cls:10s}: {cls_acc*100:5.1f}% ({count:4d} samples)")

# Classification report
print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
report = classification_report(y_test_encoded, y_pred_classes, 
                               target_names=le.classes_,
                               zero_division=0)
print(report)

# Confusion matrix
print("\nGenerating confusion matrix...")
cm = confusion_matrix(y_test_encoded, y_pred_classes)
plt.figure(figsize=(16, 14))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap='Blues', xticks_rotation='vertical', values_format='d')
plt.title(f'Confusion Matrix - Alphabet Model\nTop-1 Accuracy: {top1_accuracy*100:.2f}%', 
          fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('results/image_confusion_matrix.png', dpi=300, bbox_inches='tight')
print(" Confusion matrix saved to: results/image_confusion_matrix.png")
plt.close()

# Save metrics to file
print("\nSaving metrics to file...")
with open('results/image_metrics.txt', 'w') as f:
    f.write("IMAGE/ALPHABET MODEL EVALUATION RESULTS\n")
    f.write("="*60 + "\n\n")
    f.write(f"Model: {model_path}\n")
    f.write(f"Test samples: {len(y_test_encoded)}\n")
    f.write(f"Number of classes: {len(le.classes_)}\n")
    f.write(f"Classes: {', '.join(sorted(le.classes_))}\n\n")
    f.write(f"Top-1 Accuracy: {top1_accuracy:.4f} ({top1_accuracy*100:.2f}%)\n")
    f.write(f"Weighted F1-Score: {f1_weighted:.4f}\n")
    f.write(f"Correct/Total: {correct}/{total}\n\n")
    f.write("Per-class Accuracy:\n")
    f.write("-" * 60 + "\n")
    for cls in sorted(le.classes_):
        cls_idx = le.transform([cls])[0]
        idx = y_test_encoded == cls_idx
        if np.any(idx):
            cls_acc = (y_pred_classes[idx] == y_test_encoded[idx]).mean()
            count = np.sum(idx)
            f.write(f"  {cls:10s}: {cls_acc*100:5.1f}% ({count:4d} samples)\n")
    f.write("\n" + "="*60 + "\n")
    f.write("Classification Report:\n")
    f.write("="*60 + "\n")
    f.write(report)

print(" Metrics saved to: results/image_metrics.txt")

print("\n" + "="*60)
print("IMAGE/ALPHABET MODEL EVALUATION COMPLETE!")
print("="*60)
print(f"\nResults:")
print(f"  Accuracy: {top1_accuracy*100:.2f}%")
print(f"  F1-Score: {f1_weighted:.4f}")
print(f"  Confusion matrix: results/image_confusion_matrix.png")
print(f"  Detailed metrics: results/image_metrics.txt")
