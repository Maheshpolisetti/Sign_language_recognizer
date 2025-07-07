import numpy as np
import time
import cv2
import mediapipe as mp
import joblib
from tensorflow.keras.models import load_model # type: ignore
from collections import deque

# Load models
model_dynamic = load_model("sign_model_dynamic_mark2.h5")
le1 = joblib.load("label_encoder_dynamic.pkl")

model_static = load_model("sign_model_static_mark2.h5")
le2 = joblib.load("label_encoder.pkl")  # ['A' to 'Z', 'SPACE', 'DEL']

# Setup MediaPipe
mphands = mp.solutions.hands
hands = mphands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mpdraw = mp.solutions.drawing_utils

# Webcam
cap = cv2.VideoCapture(0)

SEQUENCE_LENGTH = 30
sequence = deque(maxlen=SEQUENCE_LENGTH)

# Display control
last_prediction_time = 0
displayed_letter = ""
display_duration = 2.0 
d_THRESHOLD = 0.85  # Confidence threshold
s_THRESHOLD = 0.8  

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    multilandmarks = results.multi_hand_landmarks

    if multilandmarks:
        for hand_landmarks in multilandmarks:
            mpdraw.draw_landmarks(img, hand_landmarks, mphands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            if len(landmarks) == 63:
                sequence.append(landmarks)

                # Predict when enough frames collected
                if len(sequence) == SEQUENCE_LENGTH:
                    current_time = time.time()
                    if current_time - last_prediction_time > display_duration:
                        seq_array = np.array(sequence).reshape(1, SEQUENCE_LENGTH, 63)
                        single_frame = np.array(landmarks).reshape(1, 63)

                        # Predict with both models
                        dyn_probs = model_dynamic.predict(seq_array)[0]
                        stat_probs = model_static.predict(single_frame)[0]

                        dyn_conf = np.max(dyn_probs)
                        stat_conf = np.max(stat_probs)

                        dyn_label = le1.inverse_transform([np.argmax(dyn_probs)])[0]
                        stat_label = le2.inverse_transform([np.argmax(stat_probs)])[0]

                        # Choose based on confidence
                        if stat_conf >= s_THRESHOLD and dyn_conf >= stat_conf:
                            displayed_letter = f"{stat_label} (S)"
                            last_prediction_time = current_time
                            sequence.clear()
                        elif dyn_conf >= d_THRESHOLD:
                            displayed_letter = f"{dyn_label} (D)"
                            last_prediction_time = current_time
                            sequence.clear()
                        
    # Display result
    if time.time() - last_prediction_time < display_duration:
        cv2.putText(img, f"Letter: {displayed_letter}", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)

    cv2.imshow("Sign Language Recognition", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
