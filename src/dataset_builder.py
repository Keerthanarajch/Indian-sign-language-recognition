import os
import cv2
import numpy as np
from tqdm import tqdm
from src.config import *
from src.feature_extraction import extract_landmarks
from src.preprocessing import global_normalize

def build_dataset():
    X_single, y_single = [], []
    X_double, y_double = [], []

    labels = sorted([d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR,d))])

    for label in labels:
        folder = os.path.join(DATA_DIR, label)
        print("Processing:", label)

        for fname in tqdm(os.listdir(folder)):
            if not fname.lower().endswith(('.jpg','.jpeg','.png')):
                continue

            img = cv2.imread(os.path.join(folder, fname))
            if img is None:
                continue

            count, hands_map = extract_landmarks(img)

            if label in single_hand_classes:
                if count != 1:
                    continue
                hand = list(hands_map.values())[0]
                X_single.append(global_normalize([hand]))
                y_single.append(label)

            elif label in double_hand_classes:
                if count != 2:
                    continue
                left = hands_map["Left"]
                right = hands_map["Right"]
                X_double.append(global_normalize([left, right]))
                y_double.append(label)

    return np.array(X_single), np.array(y_single), np.array(X_double), np.array(y_double)