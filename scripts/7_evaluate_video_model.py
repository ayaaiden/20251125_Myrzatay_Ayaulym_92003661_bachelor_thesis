import numpy as np
import pickle
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score
import matplotlib.pyplot as plt
from tensorflow import keras
import os

print("="*60)
print("EVALUATING WLASL WORD MODEL")
print("="*60)

# Create results directory
os.makedirs('results', exist_ok=True)

# ============================================================
# STEP 1: Load Test Data from Keypoints Directory
# ============================================================
print("\n[1/6] Loading test data from keypoints...")

KEYPOINTS_DIR = "data/keypoints_wlasl"

if not os.path.exists(KEYPOINTS_DIR):
    print(f"✗ Error: Directory not found: {KEYPOINTS_DIR}")
    print("\n⚠️  Please run these first:")
    print("  1. python3 scripts/0_download_wlasl_videos.py")
    print("  2. python3 scripts/15_preprocess_wlasl_videos.py")
    exit(1)

# Load all keypoints
X_list = []
y_list = []

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
            print(f" Failed to load {keypoint_file}: {e}")

X = np.array(X_list)
y = np.array(y_list)

print(f" Loaded {len(X)} sequences from {len(set(y))} words")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

if len(X) == 0:
    print("\  Error: No keypoint data found!")
    exit(1)

# ============================================================
# STEP 2: Load Model and Label Encoder
# ============================================================
print("\n[2/6] Loading model and label encoder...")

# Load model
model_path = "models/wlasl_word_model.keras"

if not os.path.exists(model_path):
    print(f" Error: Model not found: {model_path}")
    print("\n Please run first:")
    exit(1)

model = keras.models.load_model(model_path)
print(f"✓ Loaded model from: {model_path}")
print(f"Model input shape: {model.input_shape}")
print(f"Model output shape: {model.output_shape}")

# Load label encoder (NumPy array format)
labels_path = 'models/wlasl_label_encoder.npy'

if not os.path.exists(labels_path):
    print(f"✗ Error: Label encoder not found: {labels_path}")
    print("\n Please run first:")
    exit(1)

unique_words = np.load(labels_path, allow_pickle=True)
print(f" Loaded label encoder")
print(f"Number of classes: {len(unique_words)}")
print(f"Classes: {unique_words}")

# Create word-to-index mapping
word_to_idx = {word: idx for idx, word in enumerate(unique_words)}

# ============================================================
# STEP 3: Encode Labels
# ============================================================
print("\n[3/6] Encoding labels...")

# Encode all labels
y_encoded = np.array([word_to_idx[word] for word in y if word in word_to_idx])

# Filter X to match encoded y (in case some words were not in training)
X_filtered = np.array([X[i] for i, word in enumerate(y) if word in word_to_idx])

print(f" Encoded {len(y_encoded)} labels")
print(f" Filtered data shape: {X_filtered.shape}")

if len(X_filtered) == 0:
    print("\n Error: No matching data after filtering!")
    print("   Training vocabulary may not match test data.")
    exit(1)

# Use filtered data for evaluation
X_test = X_filtered
y_test_encoded = y_encoded

# ============================================================
# STEP 4: Make Predictions
# ============================================================
print("\n[4/6] Making predictions...")

y_pred = model.predict(X_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

print(f" Generated {len(y_pred_classes)} predictions")
print(f"Prediction shape: {y_pred.shape}")

# ============================================================
# STEP 5: Compute Metrics
# ============================================================
print("\n[5/6] Computing metrics...")

# Top-1 Accuracy
top1_accuracy = accuracy_score(y_test_encoded, y_pred_classes)
print(f"\n{'='*60}")
print(f"TOP-1 ACCURACY: {top1_accuracy:.4f} ({top1_accuracy*100:.2f}%)")
print(f"{'='*60}")

# Correct vs Total
correct = np.sum(y_test_encoded == y_pred_classes)
total = len(y_test_encoded)
print(f"Correct predictions: {correct}/{total}")

# F1-Score (weighted average)
f1_weighted = f1_score(y_test_encoded, y_pred_classes, average='weighted', zero_division=0)
print(f"\nWeighted F1-Score: {f1_weighted:.4f}")

# Classification Report
print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
report = classification_report(y_test_encoded, y_pred_classes, 
                               target_names=unique_words,
                               zero_division=0)
print(report)

# Per-class F1 scores
report_dict = classification_report(y_test_encoded, y_pred_classes, 
                                    target_names=unique_words,
                                    output_dict=True,
                                    zero_division=0)
f1_scores = {cls: report_dict[cls]['f1-score'] for cls in unique_words}
print("\nPer-class F1-scores:")
for cls, f1 in sorted(f1_scores.items(), key=lambda x: x[1], reverse=True):
    print(f"  {cls}: {f1:.4f}")

# ============================================================
# STEP 6: Generate and Save Confusion Matrix
# ============================================================
print("\n[6/6] Generating confusion matrix...")

cm = confusion_matrix(y_test_encoded, y_pred_classes)

plt.figure(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=unique_words)
disp.plot(cmap='Blues', xticks_rotation='vertical', values_format='d')
plt.title('Confusion Matrix - WLASL Word Model\nTop-1 Accuracy: {:.2f}%'.format(top1_accuracy*100), 
          fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('results/wlasl_confusion_matrix.png', dpi=300, bbox_inches='tight')
print(" Confusion matrix saved to: results/wlasl_confusion_matrix.png")
plt.close()

# Save metrics to file
metrics = {
    'top1_accuracy': float(top1_accuracy),
    'f1_weighted': float(f1_weighted),
    'f1_per_class': {k: float(v) for k, v in f1_scores.items()},
    'n_samples': len(X_test),
    'n_classes': len(unique_words),
    'classes': list(unique_words),
    'correct': int(correct),
    'total': int(total)
}

with open('results/wlasl_metrics.pkl', 'wb') as f:
    pickle.dump(metrics, f)
print(" Metrics saved to: results/wlasl_metrics.pkl")

# Also save as readable text file
with open('results/wlasl_metrics.txt', 'w') as f:
    f.write("WLASL WORD MODEL EVALUATION RESULTS\n")
    f.write("="*60 + "\n\n")
    f.write(f"Top-1 Accuracy: {top1_accuracy:.4f} ({top1_accuracy*100:.2f}%)\n")
    f.write(f"Weighted F1-Score: {f1_weighted:.4f}\n")
    f.write(f"Correct/Total: {correct}/{total}\n")
    f.write(f"Number of classes: {len(unique_words)}\n\n")
    f.write("Per-class F1-scores:\n")
    for cls, f1 in sorted(f1_scores.items(), key=lambda x: x[1], reverse=True):
        f.write(f"  {cls}: {f1:.4f}\n")
print(" Metrics saved to: results/wlasl_metrics.txt")

print("\n" + "="*60)
print("WLASL WORD MODEL EVALUATION COMPLETE")
print("="*60)
print("\nResults saved to:")
print("  - results/wlasl_confusion_matrix.png")
print("  - results/wlasl_metrics.pkl")
print("  - results/wlasl_metrics.txt")