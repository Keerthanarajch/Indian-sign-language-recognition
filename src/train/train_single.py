import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras import callbacks
from src.models.single_model import make_single_model

OUTPUT_DIR = "outputs/"

def train_single(X_single, y_single):
    print("Single-hand dataset shape:", X_single.shape, y_single.shape)

    if len(X_single) == 0:
        print("❌ No single-hand samples found. Skipping training.")
        return

    # Encode labels
    class_list = sorted(list(set(y_single.tolist())))
    cls2idx = {cls:i for i,cls in enumerate(class_list)}

    y_idx = np.array([cls2idx[c] for c in y_single], dtype=np.int32)

    # Split
    Xtr, Xval, ytr, yval = train_test_split(
        X_single, y_idx, test_size=0.15,
        stratify=y_idx, random_state=42
    )

    # Class weights
    weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(ytr),
        y=ytr
    )
    weights = {i:w for i,w in enumerate(weights)}

    # Model
    model = make_single_model(63, len(class_list))

    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/history", exist_ok=True)
    os.makedirs("outputs/classes", exist_ok=True)

    ckpt_path = "outputs/models/model_single.h5"

    cb = [
        callbacks.EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4),
        callbacks.ModelCheckpoint(ckpt_path, save_best_only=True, monitor="val_accuracy")
    ]

    history = model.fit(
        Xtr, ytr,
        validation_data=(Xval, yval),
        epochs=80,
        batch_size=64,
        class_weight=weights,
        callbacks=cb,
        verbose=2
    )

    # Save outputs
    with open("outputs/classes/classes_single.json", "w") as f:
        json.dump(class_list, f)

    with open("outputs/history/history_single.json", "w") as f:
        json.dump({k:[float(v) for v in vals] for k,vals in history.history.items()}, f)

    print("✅ Single-hand training complete!")