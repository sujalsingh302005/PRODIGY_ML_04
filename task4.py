"""
PRODIGY_ML_04 — Hand Gesture Recognition (CNN + SVM)
Prodigy Infotech Machine Learning Internship

Task: Develop a hand gesture recognition model that can accurately identify
      and classify different hand gestures from image or video data, enabling
      intuitive human-computer interaction and gesture-based control systems.

Dataset: https://www.kaggle.com/gti-upm/leapgestrecog
         (Download and extract — contains subfolders per gesture class)

Directory structure expected:
    leapgestrecog/
    └── 00/        (gesture 0 — Palm)
    └── 01/        (gesture 1 — L-gesture)
    └── …

Instructions:
    1. Download the dataset from the Kaggle link above
    2. Set DATASET_DIR below to the extracted root folder
    3. Run: python task04_hand_gesture_recognition.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── Image loading ─────────────────────────────
try:
    import cv2
    USE_CV2 = True
except ImportError:
    USE_CV2 = False
    try:
        from PIL import Image
    except ImportError:
        print("Install opencv-python or Pillow:  pip install opencv-python pillow")
        sys.exit(1)

# ── Sklearn ───────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay)

# ── Try TensorFlow/Keras (optional CNN) ───────
try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    from tensorflow.keras import layers, models
    from tensorflow.keras.utils import to_categorical
    USE_CNN = True
    print("✅ TensorFlow found — will train CNN model")
except ImportError:
    USE_CNN = False
    print("ℹ️  TensorFlow not found — will use SVM with HOG features")

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
DATASET_DIR  = "leapgestrecog"   # Root folder of dataset
IMG_SIZE     = (64, 64)
MAX_PER_CLASS = 200              # Limit per gesture class
BATCH_SIZE   = 32
EPOCHS       = 15

# Gesture names (LeapGestRecog dataset)
GESTURE_NAMES = {
    "00": "Palm",  "01": "L",     "02": "Fist",
    "03": "Fist (moved)", "04": "Thumb",  "05": "Index",
    "06": "Ok",    "07": "Palm (moved)", "08": "C",  "09": "Down"
}

print("=" * 60)
print("  PRODIGY_ML_04 — Hand Gesture Recognition")
print("=" * 60)


# ─────────────────────────────────────────────
def load_image(path: str) -> np.ndarray | None:
    """Load and resize image."""
    if USE_CV2:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        img = cv2.resize(img, IMG_SIZE)
    else:
        img = Image.open(path).convert("L").resize(IMG_SIZE)
        img = np.array(img)
    return img.astype(np.float32) / 255.0


# ─────────────────────────────────────────────
# 1. LOAD IMAGES
# ─────────────────────────────────────────────
images, labels = [], []
class_names_found = []

dataset_path = Path(DATASET_DIR)

if not dataset_path.exists():
    print(f"\n⚠️  '{DATASET_DIR}' not found. Generating synthetic dataset for demo…")
    np.random.seed(42)
    n_classes  = 10
    n_per_cls  = 150
    class_names_found = [GESTURE_NAMES.get(f"{i:02d}", f"Gesture_{i}") for i in range(n_classes)]
    for cls in range(n_classes):
        for _ in range(n_per_cls):
            # Create gesture-like synthetic patterns
            img = np.random.rand(*IMG_SIZE).astype(np.float32) * 0.3
            # Add unique stripe pattern per class
            freq = cls + 1
            for row in range(0, IMG_SIZE[0], freq * 4 + 4):
                img[row:row+2, :] = 1.0
            images.append(img)
            labels.append(cls)
    print(f"   Synthetic dataset: {len(images)} images, {n_classes} gesture classes")
else:
    print(f"\n📂 Loading images from '{DATASET_DIR}' …")
    class_dirs = sorted([d for d in dataset_path.iterdir() if d.is_dir()])
    for class_dir in class_dirs:
        cls_label = class_dir.name
        gesture   = GESTURE_NAMES.get(cls_label, cls_label)
        cls_images = []
        for img_path in list(class_dir.glob("**/*.png")) + \
                         list(class_dir.glob("**/*.jpg")):
            if len(cls_images) >= MAX_PER_CLASS:
                break
            img = load_image(str(img_path))
            if img is not None:
                cls_images.append(img)
        if cls_images:
            images.extend(cls_images)
            labels.extend([cls_label] * len(cls_images))
            class_names_found.append(gesture)
            print(f"   [{cls_label}] {gesture:<20} : {len(cls_images)} images")

    if len(images) == 0:
        print("❌ No images loaded. Check DATASET_DIR.")
        sys.exit(1)

# Encode labels
X_raw = np.array(images)
le    = LabelEncoder()
y     = le.fit_transform(labels)
n_cls = len(np.unique(y))

if not class_names_found:
    class_names_found = [str(c) for c in le.classes_]

print(f"\n   Total images : {len(X_raw)}")
print(f"   Classes      : {n_cls}")

# ─────────────────────────────────────────────
# 2. VISUALISE SAMPLE IMAGES
# ─────────────────────────────────────────────
fig, axes = plt.subplots(2, min(n_cls, 10), figsize=(20, 5))
fig.suptitle("Task-04 | Sample Hand Gestures per Class", fontsize=13, fontweight="bold")
shown = set()
axs = axes.flatten() if n_cls <= 10 else axes[0]
idx_map = {}
for i, lbl in enumerate(y):
    if lbl not in idx_map:
        idx_map[lbl] = i
    if len(idx_map) == n_cls:
        break
for col, (cls_id, img_idx) in enumerate(sorted(idx_map.items())):
    if col >= min(n_cls, 10):
        break
    ax = axes[0][col] if n_cls > 1 else axes
    ax.imshow(X_raw[img_idx], cmap="gray")
    lbl_name = class_names_found[col] if col < len(class_names_found) else str(cls_id)
    ax.set_title(lbl_name, fontsize=8)
    ax.axis("off")
# Second row: same but different sample
for col, (cls_id, _) in enumerate(sorted(idx_map.items())):
    if col >= min(n_cls, 10):
        break
    alt = np.where(y == cls_id)[0]
    idx2 = alt[len(alt) // 2] if len(alt) > 1 else alt[0]
    ax = axes[1][col] if n_cls > 1 else axes
    ax.imshow(X_raw[idx2], cmap="gray")
    ax.axis("off")
plt.tight_layout()
plt.savefig("task04_samples.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n📸 Sample grid saved → task04_samples.png")

# ─────────────────────────────────────────────
# 3. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=42, stratify=y)

# ─────────────────────────────────────────────
# 4A. CNN MODEL (if TensorFlow available)
# ─────────────────────────────────────────────
if USE_CNN:
    print("\n🧠 Training CNN model…")
    X_tr = X_train_raw[..., np.newaxis]   # (N, H, W, 1)
    X_te = X_test_raw[..., np.newaxis]

    y_tr_cat = to_categorical(y_train, n_cls)
    y_te_cat = to_categorical(y_test,  n_cls)

    model = models.Sequential([
        layers.Input(shape=(*IMG_SIZE, 1)),
        layers.Conv2D(32, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Conv2D(128, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(n_cls, activation="softmax"),
    ])

    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()

    history = model.fit(X_tr, y_tr_cat,
                        validation_data=(X_te, y_te_cat),
                        epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

    _, test_acc = model.evaluate(X_te, y_te_cat, verbose=0)
    y_pred = model.predict(X_te).argmax(axis=1)

    print(f"\n{'─'*40}")
    print(f"  CNN Test Accuracy : {test_acc*100:.2f}%")
    print(f"{'─'*40}")
    print(classification_report(y_test, y_pred,
                                 target_names=class_names_found[:n_cls]))

    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Task-04 | CNN Training History", fontsize=14, fontweight="bold")
    axes[0].plot(history.history["accuracy"],     label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Val")
    axes[0].set_title("Accuracy");   axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].plot(history.history["loss"],     label="Train")
    axes[1].plot(history.history["val_loss"], label="Val")
    axes[1].set_title("Loss");  axes[1].legend();  axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("task04_cnn_history.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("📊 CNN training history → task04_cnn_history.png")

    model.save("task04_gesture_cnn.keras")
    print("💾 Model saved → task04_gesture_cnn.keras")

# ─────────────────────────────────────────────
# 4B. SVM FALLBACK (always runs — good baseline)
# ─────────────────────────────────────────────
print("\n🤖 Training SVM baseline model (PCA + RBF-SVM)…")
X_flat_train = X_train_raw.reshape(len(X_train_raw), -1)
X_flat_test  = X_test_raw.reshape(len(X_test_raw),  -1)

n_comp = min(150, X_flat_train.shape[0] - 1, X_flat_train.shape[1])
svm_pipe_steps = [
    ("scaler", StandardScaler()),
    ("pca",    PCA(n_components=n_comp, whiten=True, random_state=42)),
    ("svm",    SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)),
]
from sklearn.pipeline import Pipeline
svm_pipe = Pipeline(svm_pipe_steps)
svm_pipe.fit(X_flat_train, y_train)

y_pred_svm = svm_pipe.predict(X_flat_test)
svm_acc    = accuracy_score(y_test, y_pred_svm)

print(f"\n{'─'*40}")
print(f"  SVM Accuracy : {svm_acc*100:.2f}%")
print(f"{'─'*40}")
print(classification_report(y_test, y_pred_svm,
                             target_names=class_names_found[:n_cls]))

# ─────────────────────────────────────────────
# 5. CONFUSION MATRIX
# ─────────────────────────────────────────────
best_pred  = y_pred if USE_CNN else y_pred_svm
best_label = "CNN" if USE_CNN else "SVM"

cm   = confusion_matrix(y_test, best_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=class_names_found[:n_cls])
fig, ax = plt.subplots(figsize=(12, 10))
disp.plot(ax=ax, colorbar=True, cmap="Blues", xticks_rotation=45)
acc = test_acc if USE_CNN else svm_acc
ax.set_title(f"Task-04 | {best_label} Confusion Matrix  (Acc={acc*100:.1f}%)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("task04_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
print("📊 Confusion matrix saved → task04_confusion_matrix.png")

print("\n✅ Task-04 complete!\n")
