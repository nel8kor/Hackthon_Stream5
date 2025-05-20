#!/usr/bin/env py
# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import mediapipe as mp
from collections import Counter
import pandas as pd
from keras.utils import to_categorical # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore
from tqdm import tqdm
import traceback

# Constants
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
EAR_THRESHOLD = 0.25
PITCH_THRESHOLD = -6
FRAME_SIZE = (120, 160)  # Resize frames to 112x112

FACE_3D_POINTS = np.array([
    (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
])

def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def eye_aspect_ratio(eye_pts):
    A = euclidean(eye_pts[1], eye_pts[5])
    B = euclidean(eye_pts[2], eye_pts[4])
    C = euclidean(eye_pts[0], eye_pts[3])
    return (A + B) / (2.0 * C)

def estimate_pitch(landmarks, w, h):
    try:
        image_points = np.array([
            [landmarks[1].x * w, landmarks[1].y * h],
            [landmarks[152].x * w, landmarks[152].y * h],
            [landmarks[263].x * w, landmarks[263].y * h],
            [landmarks[33].x * w, landmarks[33].y * h],
            [landmarks[287].x * w, landmarks[287].y * h],
            [landmarks[57].x * w, landmarks[57].y * h]
        ], dtype="double")

        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")

        _, rot_vec, _ = cv2.solvePnP(FACE_3D_POINTS, image_points, camera_matrix, np.zeros((4, 1)))
        rot_mat, _ = cv2.Rodrigues(rot_vec)
        pitch, *_ = cv2.RQDecomp3x3(rot_mat)
        return pitch[0]
    except Exception:
        traceback.print_exc()
        return None

def extract_frames(video_path):
    frames, labels = [], []
    drowsy_counter = 0

    cap = cv2.VideoCapture(video_path)
    FPS = cap.get(cv2.CAP_PROP_FPS)
    DISTRACTED_FRAMES_THRESHOLD = int(0.5 * FPS)
    DROWSY_FRAMES_THRESHOLD = int(1.0 * FPS)

    with mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=100)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            if not result.multi_face_landmarks:
                continue

            landmarks = result.multi_face_landmarks[0].landmark
            left_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in LEFT_EYE]
            right_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in RIGHT_EYE]
            ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

            pitch = estimate_pitch(landmarks, w, h)
            if pitch is None:
                continue

            if ear < EAR_THRESHOLD:
                drowsy_counter += 1
                if drowsy_counter >= DROWSY_FRAMES_THRESHOLD and pitch < PITCH_THRESHOLD:
                    label = 2  # drowsy
                elif drowsy_counter >= DISTRACTED_FRAMES_THRESHOLD:
                    label = 1  # distracted
                else:
                    label = 0
            else:
                drowsy_counter = 0
                label = 0

            resized = cv2.resize(frame, FRAME_SIZE)
            frames.append(img_to_array(resized / 255.0))
            labels.append(label)

            print(f"[INFO] EAR: {ear:.2f}, PITCH: {pitch:.2f}, LABEL: {label}")

    cap.release()
    return np.array(frames), np.array(labels), FPS

def process_dataset(dataset_dir="dataset/val"):
    X, y, stats = [], [], []

    for folder in tqdm(os.listdir(dataset_dir), desc="Processing Folders"):
        folder_path = os.path.join(dataset_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        for video_file in tqdm(os.listdir(folder_path), desc=f"Videos in {folder}", leave=False):
            if not video_file.lower().endswith(('.mp4', '.avi', '.mov')):
                continue

            path = os.path.join(folder_path, video_file)
            frames, labels, fps = extract_frames(path)

            if frames.size == 0:
                continue

            X.extend(frames)
            y.extend(labels)

            c = Counter(labels)
            stats.append({
                "video_path": path,
                "total_frames": len(labels),
                "awake": c[0],
                "distracted": c[1],
                "drowsy": c[2],
                "FPS": fps
            })

    return np.array(X), np.array(y), pd.DataFrame(stats)

# --- Run the pipeline ---
X, y, df_stats = process_dataset()
# Save stats
df_stats.to_excel("video_label_summary.xlsx", index=False)
print("X shape:", X.shape)
print("y shape:", y.shape)
print("X dtype:", X.dtype)
print("y dtype:", y.dtype)

#print awake, distracted and drowsy samples before filter
print(f"awake samples: {(y == 0).sum()}")
print(f"distracted samples: {(y == 1).sum()}")
print(f"drowsy samples: {(y == 2).sum()}")

# Save the filtered arrays
np.save("X_val.npy", X)
np.save("y_val.npy", y)
print("Saved filtered data as X.npy and y.npy")