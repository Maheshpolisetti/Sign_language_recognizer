# Sign Language Recognizer

## Introduction
An integrated machine learning system that recognizes both static ASL (American Sign Language) alphabets and dynamic hand gestures using computer vision and deep learning techniques. Designed for real-time sign language interpretation to improve accessibility and communication.

## Objective
To build a robust, real-time sign language interpretation tool that can detect and classify both static signs (like alphabets) and dynamic gestures (like "hello", "thank you", "bye") using camera input, combining classical computer vision with deep learning models.

## Features
- Real-time Hand Tracking using MediaPipe and OpenCV.  
- Static ASL Alphabet Classification with over 97.6% accuracy using a custom-trained CNN.  
- Dynamic Gesture Recognition using LSTM models on temporal hand landmark sequences, achieving 96.5% accuracy.  
- Training pipeline with:  
  - Learning Rate Scheduler (`ReduceLROnPlateau`)  
  - Early Stopping to prevent overfitting  
  - Frame-wise preprocessing and data augmentation  
- Integrated both models for real-time end-to-end prediction from webcam input.

## Methodology

### 1. Data Collection & Labeling
- Collected real-time video data using webcam and MediaPipe hand landmarks.  
- Two types of data:  
  - **Static signs** → one frame per label (ASL alphabets A-Z).  
  - **Dynamic gestures** → sequences of frames per gesture (e.g., hello, bye, thanks).  
- Converted hand landmarks (21 keypoints × (x, y, z)) into NumPy arrays.

### 2. Preprocessing
- Ensured class balance via undersampling/oversampling.  
- Applied temporal data augmentation techniques to improve generalization:  
  - Dropped frames within sequences to simulate motion variation.  
  - Added small Gaussian noise to landmark coordinates.  
- Used fixed-length sliding window (e.g., 30 frames) for gesture sequences; padded or truncated as needed.  
- Ensured strict separation between training and validation data to avoid data leakage:  
  - Used different participants and sequences in validation set.  
  - Avoided overlap of samples (or augmented samples) between training and validation splits.

### 3. Model Architecture

**Static Sign Model (ASL Alphabet)**  
- **Model type:** Dense Neural Network (DNN)  
- **Input shape:** (21 landmarks × 3 coords)  
- **Layers:**  
  - Dense(128, ReLU) → Dropout(0.3)  
  - Dense(64, ReLU) → Dropout(0.3)  
  - Dense(28, Softmax) → for 26 ASL letters + 2 [del, space]  

**Dynamic Gesture Model**  
- **Model type:** LSTM (Long Short-Term Memory)  
- **Input shape:** (30 timesteps × 63 features)  
- **Layers:**  
  - LSTM(64) → Dropout(0.3)  
  - Dense(64, ReLU)  
  - Dense(5, Softmax) → for gestures: hello, bye, thanks, drink, love

### 4. Training
- **Optimizer:** Adam  
- **Loss:** categorical_crossentropy  
- **Learning rate scheduler:** ReduceLROnPlateau  
- **Early stopping** based on validation loss  
- **Accuracy:**  
  - Static model: **97.63%**  
  - Dynamic model: **96.53%**

### 5. Inference Pipeline
- Captures video in real time using OpenCV.  
- Extracts hand landmarks using MediaPipe.  
- Passes input to:  
  - Static classifier (single frame)  
  - OR dynamic classifier (sequence buffer)  
- Displays prediction as text on screen (no audio output).

## Demo Video
_(Insert link or embed once available)_

## Future Improvements
- Deploy as a web app using Streamlit or Flask.  
- Add support for two-hand signs and facial expressions using holistic MediaPipe pipeline.
