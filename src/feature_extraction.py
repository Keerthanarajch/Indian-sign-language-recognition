import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)

def extract_landmarks(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    if not res.multi_hand_landmarks:
        return 0, {}

    out = {}
    for i, hl in enumerate(res.multi_hand_landmarks):
        label = res.multi_handedness[i].classification[0].label
        coords = np.array([[lm.x, lm.y, lm.z] for lm in hl.landmark])
        out[label] = coords

    return len(out), out