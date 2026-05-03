import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import json, time, os
from collections import deque, Counter
from gtts import gTTS
import threading, tempfile

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
MODEL_SINGLE = "model_single_final.h5"
CLASSES_SINGLE = "classes_single_simple.json"

MODEL_DOUBLE = "model_double_final.h5"
CLASSES_DOUBLE = "classes_double_simple.json"

FLIP_HORIZONTAL = True
CONF_THRESHOLD = 0.65
BUFFER_SIZE = 12
MIN_STABLE = 6
COOLDOWN = 1.8

# ---------------------------------------------------------
# LOAD MODELS + CLASSES
# ---------------------------------------------------------
def load_model_and_classes(model_path, classes_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file missing: {model_path}")
    if not os.path.exists(classes_path):
        raise FileNotFoundError(f"Classes file missing: {classes_path}")

    model = tf.keras.models.load_model(model_path)
    with open(classes_path) as f:
        classes = json.load(f)
    return model, classes

print("Loading models...")
model_single, classes_single = load_model_and_classes(MODEL_SINGLE, CLASSES_SINGLE)
model_double, classes_double = load_model_and_classes(MODEL_DOUBLE, CLASSES_DOUBLE)
print("Models loaded successfully.\n")

# ---------------------------------------------------------
# MEDIAPIPE
# ---------------------------------------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)

# ---------------------------------------------------------
# LANDMARK SMOOTHER
# ---------------------------------------------------------
class LandmarkSmoother:
    def __init__(self, alpha=0.6):
        self.alpha = alpha
        self.prev = None

    def smooth(self, arr):
        arr = np.array(arr, dtype=np.float32)
        if self.prev is None:
            self.prev = arr
            return arr
        self.prev = self.alpha * arr + (1 - self.alpha) * self.prev
        return self.prev

smoother = LandmarkSmoother()

# ---------------------------------------------------------
# GLOBAL NORMALIZATION (fixed)
# ---------------------------------------------------------
def global_normalize(arrs):
    coords = np.vstack(arrs).astype(np.float32)
    mean = coords.mean(axis=0)
    coords -= mean
    span = max(coords[:,0].ptp(), coords[:,1].ptp(), 1e-6)
    coords /= span
    coords -= coords.mean(axis=0)
    return coords.flatten()

# ---------------------------------------------------------
# FEATURE EXTRACTION
# ---------------------------------------------------------
def prepare_single_feature(hand_lm):
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_lm.landmark])
    pts = smoother.smooth(pts)
    return global_normalize([pts])

def prepare_double_feature(left_lm, right_lm):
    L = np.array([[lm.x, lm.y, lm.z] for lm in left_lm.landmark])
    R = np.array([[lm.x, lm.y, lm.z] for lm in right_lm.landmark])
    L = smoother.smooth(L)
    R = smoother.smooth(R)
    return global_normalize([L, R])

# ---------------------------------------------------------
# NON-BLOCKING TTS
# ---------------------------------------------------------
def speak_label(text):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    path = tmp.name
    tmp.close()

    gTTS(text=text, lang="en").save(path)

    def play(path):
        try:
            from playsound import playsound
            playsound(path)
        except:
            pass
        finally:
            try: os.remove(path)
            except: pass

    threading.Thread(target=play, args=(path,), daemon=True).start()

# ---------------------------------------------------------
# STABILITY BUFFER
# ---------------------------------------------------------
pred_buffer = deque(maxlen=BUFFER_SIZE)

def stable_prediction():
    if not pred_buffer:
        return None
    idx, count = Counter(pred_buffer).most_common(1)[0]
    return idx if count >= MIN_STABLE else None

# ---------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

print("Starting real-time ISL detection... Press 'q' to quit.\n")

last_spoken = None
last_spoken_time = 0
assembled_text = ""

cv2.namedWindow("ISL → Text & Speech", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ISL → Text & Speech", 900, 650)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    if FLIP_HORIZONTAL:
        frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)

    hand_count = 0
    pred_label = "_"
    conf = 0.0

    left_lm, right_lm = None, None

    if res.multi_hand_landmarks:
        hand_count = len(res.multi_hand_landmarks)

        for i, hl in enumerate(res.multi_hand_landmarks):
            label = res.multi_handedness[i].classification[0].label
            if label == "Left": left_lm = hl
            if label == "Right": right_lm = hl
            mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

        # ----------------------------
        # SINGLE HAND
        # ----------------------------
        if hand_count == 1:
            only = left_lm if left_lm else right_lm
            vec = prepare_single_feature(only)
            probs = model_single.predict(vec.reshape(1, -1), verbose=0)[0]
            idx = int(np.argmax(probs))
            pred_label = classes_single[idx]
            conf = float(probs[idx])

        # ----------------------------
        # DOUBLE HAND
        # ----------------------------
        elif hand_count == 2:
            if left_lm is None or right_lm is None:
                sorted_lm = sorted(
                    res.multi_hand_landmarks,
                    key=lambda lm: np.mean([p.x for p in lm.landmark])
                )
                left_lm, right_lm = sorted_lm

            vec = prepare_double_feature(left_lm, right_lm)
            probs = model_double.predict(vec.reshape(1, -1), verbose=0)[0]
            idx = int(np.argmax(probs))
            pred_label = classes_double[idx]
            conf = float(probs[idx])

    else:
        pred_buffer.clear()

    # ---------------------------------------------------------
    # STABILITY & TTS
    # ---------------------------------------------------------
    if hand_count > 0:
        pred_buffer.append(pred_label)
        stable = stable_prediction()
        now = time.time()

        if stable and conf >= CONF_THRESHOLD:
            if stable != last_spoken and (now - last_spoken_time) > COOLDOWN:
                speak_label(stable)
                assembled_text += stable
                last_spoken = stable
                last_spoken_time = now

    # ---------------------------------------------------------
    # HUD
    # ---------------------------------------------------------
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (260, frame.shape[0]), (0,0,0), -1)
    frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

    cv2.putText(frame, f"Hands: {hand_count}", (20,50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    cv2.putText(frame, f"Current: {pred_label} ({conf:.2f})", (20,100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,255,255), 2)
    cv2.putText(frame, "Text: " + assembled_text[-25:], (20,150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,255,200), 2)

    cv2.imshow("ISL → Text & Speech", frame)

    key = cv2.waitKey(5)
    if key == ord('q'):
        break
    if key == ord('c'):
        assembled_text = ""

cap.release()
cv2.destroyAllWindows()
