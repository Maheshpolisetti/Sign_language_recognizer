import cv2
import mediapipe as mp
import pandas as pd
import os

# Root folder containing subfolders A/, B/, ..., Z/
IMAGE_FOLDER = r"\asl_alphabet_train" # download your dataset form kaggle and paste that link here for training 
MAX_IMAGES_PER_CLASS = 7000

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

data = []

# Loop through each class folder
for label in sorted(os.listdir(IMAGE_FOLDER)):
    label_folder = os.path.join(IMAGE_FOLDER, label)
    if not os.path.isdir(label_folder):
        continue  # skip non-folder items

    processed_count = 0
    print(f"Processing label: {label}")

    # Going through images in the label folder
    for file in sorted(os.listdir(label_folder)):
        if not file.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        if processed_count >= MAX_IMAGES_PER_CLASS:
            print(f"Reached 7000 images for label: {label}, skipping remaining.")
            break

        path = os.path.join(label_folder, file)
        image = cv2.imread(path)
        if image is None:
            print(f"[Warning] Couldn't read {file}, skipping.") # skipping images that can't find landmarks with mediapipe
            continue

        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # changing into RBG to use with Mediapipe
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            landmarks.append(label.upper())
            data.append(landmarks)
            processed_count += 1
        else:
            print(f"[Info] No hand detected in {file}")

    print(f"{label}: Processed {processed_count} images\n")

# saving labels as file name 
columns = [f'{axis}{i}' for i in range(21) for axis in ['x', 'y', 'z']] + ['label']
df = pd.DataFrame(data, columns=columns)

# Save to CSV
df.to_csv('sign_landmarks.csv', index=False)
print("Saved balanced landmark data to sign_landmarks_7000max.csv")
