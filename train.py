import os
import numpy as np
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Paths

# Robust: Always use workspace root for data path
WORKSPACE_ROOT = Path(__file__).resolve().parent
DATA_ROOT = WORKSPACE_ROOT / "data" / "raw" / "Railway Track fault Detection Updated" / "Train"
DEFECTIVE = DATA_ROOT / "Defective"
NON_DEFECTIVE = DATA_ROOT / "Non defective"
if not DEFECTIVE.exists() or not NON_DEFECTIVE.exists():
    raise RuntimeError(f"Could not find expected folders: {DEFECTIVE} or {NON_DEFECTIVE}")

IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 50
KFOLDS = 5

# Data loading

def load_images_labels(def_dir, nondef_dir):
    images, labels = [], []
    for f in os.listdir(def_dir):
        img = cv2.imread(str(def_dir / f), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(1)
    for f in os.listdir(nondef_dir):
        img = cv2.imread(str(nondef_dir / f), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(0)
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

# Data augmentation
aug_gen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

def augment(images, labels, n_aug=3):
    aug_imgs, aug_lbls = [], []
    # images shape: (N, 128, 128, 3) expected
    for i, (img, lbl) in enumerate(zip(images, labels)):
        # img shape: (128, 128, 3)
        img_batch = np.expand_dims(img, 0)  # (1, 128, 128, 3)
        it = aug_gen.flow(img_batch, batch_size=1)
        for _ in range(n_aug):
            batch = next(it)
            aug_imgs.append(batch[0].astype(np.uint8))
            aug_lbls.append(lbl)
    return np.array(aug_imgs), np.array(aug_lbls)

def build_model(input_shape):
    base = VGG16(include_top=False, input_shape=input_shape, weights='imagenet')
    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = Dropout(0.5)(x)
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base.input, outputs=out)
    for layer in base.layers[:-4]:
        layer.trainable = False
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def main():
    print(f"Loading data from {DEFECTIVE} and {NON_DEFECTIVE}")
    images, labels = load_images_labels(DEFECTIVE, NON_DEFECTIVE)
    images = images / 255.0
    # Convert grayscale to RGB (N, 128, 128, 3)
    images = np.expand_dims(images, -1)
    images = np.repeat(images, 3, axis=-1)
    # Augment
    aug_imgs, aug_lbls = augment(images, labels, n_aug=2)
    X = np.concatenate([images, aug_imgs])
    y = np.concatenate([labels, aug_lbls])
    print(f"Total images after augmentation: {X.shape[0]}")
    # KFold
    kf = KFold(n_splits=KFOLDS, shuffle=True, random_state=42)
    val_accuracies = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
        print(f"\nFold {fold}/{KFOLDS}")
        model = build_model(X.shape[1:])
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
            ModelCheckpoint(f'best_model_fold{fold}.keras', save_best_only=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6)
        ]
        h = model.fit(
            X[train_idx], y[train_idx],
            validation_data=(X[val_idx], y[val_idx]),
            epochs=EPOCHS, batch_size=BATCH_SIZE,
            callbacks=callbacks, verbose=2
        )
        val_preds = (model.predict(X[val_idx]) > 0.5).astype(int)
        acc = accuracy_score(y[val_idx], val_preds)
        print(f"Validation accuracy (fold {fold}): {acc:.4f}")
        val_accuracies.append(acc)
    print(f"\nMean validation accuracy: {np.mean(val_accuracies):.4f}")
    print("Training complete. Best models saved as best_model_foldN.keras.")

if __name__ == "__main__":
    main()
