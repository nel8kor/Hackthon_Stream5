#!/usr/bin/env py
# -*- coding: utf-8 -*-
import os
import cv2 # type: ignore
import numpy as np # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tqdm import tqdm # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.layers import GlobalAveragePooling2D # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore

def extract_frames(video_path, frame_size=(64, 64), max_frames=30, skip_rate=5):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret or len(frames) >= max_frames:
            break
        if count % skip_rate == 0:
            frame = cv2.resize(frame, frame_size)
            frame = frame / 255.0
            frame = img_to_array(frame)
            frames.append(frame)
        count += 1

    cap.release()
    return np.array(frames)

def video_generator(data_dir, label_map, batch_size=32, frame_size=(64, 64), max_frames=30):
    video_files = []
    labels = []

    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path) or folder not in label_map:
            continue
        for file in os.listdir(folder_path):
            if file.endswith(('.mp4', '.avi', '.mov')):
                video_files.append(os.path.join(folder_path, file))
                labels.append(label_map[folder])

    index = 0
    while True:
        batch_X, batch_y = [], []
        for _ in range(batch_size):
            if index >= len(video_files):
                index = 0

            video_path = video_files[index]
            label = labels[index]
            frames = extract_frames(video_path, frame_size, max_frames)

            # Optional: use one frame or average/stack
            if frames.shape[0] > 0:
                # Example: use the middle frame for single-frame CNN
                middle_frame = frames[len(frames) // 2]
                batch_X.append(middle_frame)
                batch_y.append(label)
            index += 1

        yield np.array(batch_X), to_categorical(np.array(batch_y), num_classes=len(label_map))

# Dataset loading with progress
X, y = [], []
#Training the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(),
    GlobalAveragePooling2D(),  # replaces Flatten safely
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
label_map = {'awake': 0, 'drowsy': 1}
train_gen = video_generator('dataset/train', label_map, batch_size=32, frame_size=(240, 320))
val_gen   = video_generator('dataset/val', label_map, batch_size=32, frame_size=(240, 320))
X, y = next(train_gen)  # Fetch the first batch of data
X_val, y_val = next(val_gen)
# Save dataset for reuse
np.save('X.npy', X)
np.save('y.npy', y)
np.save('X_val.npy', X_val)
np.save('y_val.npy', y_val)
print(f"Saved X shape: {X.shape}, y shape: {y.shape}")
print(f"Saved X shape: {X_val.shape}, y shape: {y_val.shape}")

history = model.fit(
    train_gen,
    steps_per_epoch=100,
    validation_data=val_gen,
    validation_steps=20,
    epochs=10
)
#Save the model
model.save('model.h5')

plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')

plt.title('Training and Validation Accuracy & Loss')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()
