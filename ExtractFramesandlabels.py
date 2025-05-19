#!/usr/bin/env py
# -*- coding: utf-8 -*-
import os
import cv2 # type: ignore
import numpy as np # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tqdm import tqdm # type: ignore
import mediapipe as mp
from collections import Counter
import pandas as pd
from openpyxl.workbook import Workbook

# Define EAR function
def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def eye_aspect_ratio(eye_pts):
    A = euclidean(eye_pts[1], eye_pts[5])
    B = euclidean(eye_pts[2], eye_pts[4])
    C = euclidean(eye_pts[0], eye_pts[3])
    return (A + B) / (2.0 * C)

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
EAR_THRESHOLD = 0.26
PITCH_THRESHOLD=-6
FPS = 8
DISTRACTED_FRAMES_THRESHOLD = int(0.5 * FPS)  # 5
DROWSY_FRAMES_THRESHOLD = FPS               # 10


# Head pose 3D model points
FACE_3D_POINTS = np.array([
    (0.0, 0.0, 0.0),         # Nose tip
    (0.0, -330.0, -65.0),    # Chin
    (-225.0, 170.0, -135.0), # Left eye corner
    (225.0, 170.0, -135.0),  # Right eye corner
    (-150.0, -150.0, -125.0),# Left mouth corner
    (150.0, -150.0, -125.0)  # Right mouth corner
])

def extract_frames_from_video(video_path, size=(240, 320)):
    drowsy_counter = 0
    distracted_counter = 0
    valid_frames=0
    label = 0  # awake
    

    cap = cv2.VideoCapture(video_path)
    frames = []
    labels = []

    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=100)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            if result.multi_face_landmarks:
                valid_frames += 1
                landmarks = result.multi_face_landmarks[0]
                left_eye = [(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in LEFT_EYE]
                right_eye = [(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in RIGHT_EYE]

                ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0

                # Head pose
                try:
                    image_points = np.array([
                        [landmarks.landmark[1].x * w, landmarks.landmark[1].y * h],
                        [landmarks.landmark[152].x * w, landmarks.landmark[152].y * h],
                        [landmarks.landmark[263].x * w, landmarks.landmark[263].y * h],
                        [landmarks.landmark[33].x * w, landmarks.landmark[33].y * h],
                        [landmarks.landmark[287].x * w, landmarks.landmark[287].y * h],
                        [landmarks.landmark[57].x * w, landmarks.landmark[57].y * h],
                    ], dtype="double")

                    focal_length = w
                    center = (w / 2, h / 2)
                    camera_matrix = np.array([
                        [focal_length, 0, center[0]],
                        [0, focal_length, center[1]],
                        [0, 0, 1]
                    ], dtype="double")
                    dist_coeffs = np.zeros((4, 1))

                    _, rot_vec, _ = cv2.solvePnP(FACE_3D_POINTS, image_points, camera_matrix, dist_coeffs)
                    rot_mat, _ = cv2.Rodrigues(rot_vec)
                    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rot_mat)
                    pitch = angles[0]

                    if ear < EAR_THRESHOLD and pitch < PITCH_THRESHOLD:
                        drowsy_counter += 1
                        distracted_counter = 0  # reset
                        if drowsy_counter >= DROWSY_FRAMES_THRESHOLD:
                            label = 2  # drowsy
                        else:
                            label = 0

                    elif ear < EAR_THRESHOLD:
                        distracted_counter += 1
                        drowsy_counter = 0  # reset
                        if distracted_counter >= DISTRACTED_FRAMES_THRESHOLD:
                            label = 1  # distracted
                        else:
                            label = 0

                    else:
                        drowsy_counter = 0
                        distracted_counter = 0
                        label = 0  # awake



                    resized = cv2.resize(frame, size)
                    frames.append(img_to_array(resized / 255.0))

                    labels.append(label)

                    print(f"EAR: {ear:.2f}, Pitch: {pitch:.2f}, Label: {label}, DrowsyC: {drowsy_counter}, DistractedC: {distracted_counter}")


                except Exception as e:
                    print(f"[WARN] Head pose estimation failed: {e}")
                    continue

    cap.release()

    if valid_frames == 0:
        print(f"Warning: No face detected in {video_path}")
        return None, None

    return np.array(frames), np.array(labels)

# Dataset loading with progress
X, y = [], []
video_stats = []

for folder in tqdm(os.listdir(os.path.join("dataset", "val")), desc="Folders"):
    folder_path = os.path.join(os.path.join("dataset", "val"), folder)
    if not os.path.isdir(folder_path):
        continue
    # label = label_map.get(folder)
    # if label is None:
    #     print(f"Warning: Folder '{folder}' not in label_map. Skipping.")
    #     continue
 
    for file in tqdm(os.listdir(folder_path), desc=f"Videos in {folder}", leave=False):
        if not file.endswith(('.mp4', '.avi', '.mov')):
            continue
        video_path = os.path.join(folder_path, file)
        frames, labels = extract_frames_from_video(video_path)
        X.extend(frames)
        y.extend(labels)

        counts = Counter(labels)
        print(f"{video_path} -> Total: {len(labels)}, Awake: {counts[0]}, Distracted: {counts[1]}, Drowsy: {counts[2]}")
        video_stats.append({
        "video_path": video_path,
        "total_frames": len(labels),
        "awake": counts[0],
        "destruction": counts[1],
        "drowsy": counts[2]
        })
        
df_stats = pd.DataFrame(video_stats)
df_stats.to_excel("video_label_summary.xlsx", index=False)
print("Excel summary saved as video_label_summary.xlsx")
 
# Convert to numpy arrays
X = np.array(X)
y = to_categorical(np.array(y), num_classes=3)
label_counts = np.argmax(y, axis=1)
total = len(label_counts)
print("Awake samples:", np.sum(label_counts == 0))
print("Distracted samples:", np.sum(label_counts == 1))
print("Drowsy samples:", np.sum(label_counts == 2))

 
# Save dataset for reuse
print("...........saving in progress..........")
np.save('X.val', X)
np.save('y.val', y)
print(".............saved..............")
print(f"Saved X shape: {X.shape}, y shape: {y.shape}")