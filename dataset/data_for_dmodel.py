import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# Initialize Mediapipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Sequence data store
data = []

# for data extraction from live data change this label ["Hello","Bye","Love","Thanks","Drink"]
label="labels"

# to handle sequence start and stop and quiting webcam
print("Press 'c' to start recording, 's' to stop recording, 'q' to quit.")

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # to work with mediapipe we need RGB images 
    results = hands.process(img_rgb)
    multi_landmarks = results.multi_hand_landmarks

    # Draw current frame landmarks
    if multi_landmarks:
        for hand_landmarks in multi_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("sign recording", img)

    key = cv2.waitKey(10) & 0xFF

    # to Start capturing sequence (key==c)
    if key == ord("c"):
        print("Recording... Press 's' to stop.")
        curr_seq = []

        while True:
            success, img = cap.read()
            if not success:
                break
            img = cv2.flip(img, 1)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            multi_landmarks = results.multi_hand_landmarks

            if multi_landmarks:
                hand_data = []
                for lm in multi_landmarks[0].landmark:  # using only one hand
                    hand_data.extend([lm.x, lm.y, lm.z])
                curr_seq.append(hand_data)
                mp_draw.draw_landmarks(img, multi_landmarks[0], mp_hands.HAND_CONNECTIONS)

            cv2.imshow("sign recording", img)
            key = cv2.waitKey(10) & 0xFF
            if key == ord("s"):
                print(f"Recorded {len(curr_seq)} frames.")
                if len(curr_seq)>=15: # avoiding short sequence 
                    data.append(curr_seq)
                else:
                    print("not saved due to short seq")
                break

    elif key==ord("q"):
        break

# destory recorded frames 
cap.release()
cv2.destroyAllWindows()

# data augmentation
def add_noise(seq,sigma=0.01):
    seq=np.array(seq)
    noise = np.random.normal(0, sigma, seq.shape)
    return (seq + noise).tolist()

def drop_frames(seq, drop_rate=0.1):
    n = len(seq)
    keep = int(n*(1-drop_rate)) # removing 10 % frames 
    if keep < 1:
        return seq  # avoid empty
    indices = sorted(np.random.choice(n, keep, replace=False))
    return list(seq[i] for i in indices)

def normalize_sequence_length(sequence, target_len=30): # making all sequence 30 frames 
    seq_len = len(sequence)

    if seq_len == target_len:
        return sequence
    elif seq_len > target_len:
        start = (seq_len - target_len) // 2 # if we have more then 30 frames then removing frames form start and end equally 
        return sequence[start:start+target_len]
    else:
        # Pad with last frames to get 30 frames
        last = sequence[-1]
        sequence = sequence + [last.copy() for _ in range(target_len - len(sequence))]
        return sequence

def augment_and_normalize(sequence, label, augmentations=5): # augment each inputs into 5 sequence 
    augmented_data = []

    for _ in range(augmentations):
        aug_seq = add_noise(sequence, sigma=0.01)
        aug_seq = drop_frames(aug_seq, drop_rate=0.1)
        aug_seq = normalize_sequence_length(aug_seq, target_len=30)
        augmented_data.append({
            "sequence": aug_seq,
            "label": label
        })

    return augmented_data

all_augmented = []
for seq in data:
    all_augmented.extend(augment_and_normalize(seq, label, augmentations=10))

import json

# Save each sequence and labels as JSON
df = pd.DataFrame({
    "sequence": [json.dumps(d["sequence"]) for d in all_augmented],
    "label": [d["label"] for d in all_augmented]
})
# df.to_csv(f"seq_data_{label}.csv", index=False) # adding new files 
df.to_csv(f"seq_data_{label}.csv", mode='a', header=False, index=False) # appending into files
print(f"Sequence data saved to seq_data_{label}.csv.") 