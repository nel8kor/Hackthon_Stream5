#!/usr/bin/env py
# -*- coding: utf-8 -*-
import os
import cv2 # type: ignore
import numpy as np # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tqdm import tqdm # type: ignore

label_map = {'awake': 0, 'drowsy': 1}

def extract_frames_from_video(video_path, label, size=(240, 320)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    labels = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, size)
        frame = frame / 255.0  # Normalize
        frames.append(img_to_array(frame))
        labels.append(label)
    cap.release()
    return np.array(frames), np.array(labels)
 
# Dataset loading with progress
X_Val, y_Val = [], []
 
for folder in tqdm(os.listdir("train"), desc="Folders"):
    folder_path = os.path.join("train", folder)
    if not os.path.isdir(folder_path):
        continue
    label = label_map.get(folder)
    if label is None:
        print(f"Warning: Folder '{folder}' not in label_map. Skipping.")
        continue
 
    for file in tqdm(os.listdir(folder_path), desc=f"Videos in {folder}", leave=False):
        if not file.endswith(('.mp4', '.avi', '.mov')):
            continue
        video_path = os.path.join(folder_path, file)
        frames, labels = extract_frames_from_video(video_path, label)
        X_Val.extend(frames)
        y_Val.extend(labels)
 
# Convert to numpy arrays
X_Val = np.array(X_Val)
y_Val = to_categorical(np.array(y_Val), num_classes=2)
 
# Save dataset for reuse
np.save('X_Val.npy', X_Val)
np.save('y_Val.npy', y_Val)
print(f"Saved X shape: {X_Val.shape}, y shape: {y_Val.shape}")