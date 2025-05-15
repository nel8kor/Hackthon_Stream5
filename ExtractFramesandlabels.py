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
X, y = [], []
 
for folder in tqdm(os.listdir("dataset/train"), desc="Folders"):
    folder_path = os.path.join("dataset/train", folder)
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
        X.extend(frames)
        y.extend(labels)
 
# Convert to numpy arrays
X = np.array(X)
y = to_categorical(np.array(y).astype('float16'), num_classes=2)
 
# Save dataset for reuse
print("...........saving in progress..........")
np.save('X.npy', X)
np.save('y.npy', y)
print(".............saved..............")
print(f"Saved X shape: {X.shape}, y shape: {y.shape}")