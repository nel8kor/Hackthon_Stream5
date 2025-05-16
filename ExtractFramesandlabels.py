#!/usr/bin/env py
# -*- coding: utf-8 -*-
import os
import cv2 # type: ignore
import numpy as np # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from tqdm import tqdm # type: ignore
import mediapipe as mp

label_map = {'awake': 0, 'drowsy': 1}

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
EAR_THRESHOLD = 0.25
FPS = 10
CONSEC_FRAMES = int(2 * FPS)

# Head pose 3D model points
FACE_3D_POINTS = np.array([
    (0.0, 0.0, 0.0),         # Nose tip
    (0.0, -330.0, -65.0),    # Chin
    (-225.0, 170.0, -135.0), # Left eye corner
    (225.0, 170.0, -135.0),  # Right eye corner
    (-150.0, -150.0, -125.0),# Left mouth corner
    (150.0, -150.0, -125.0)  # Right mouth corner
])

def extract_frames_from_video(video_path, label, size=(240, 320)):
    CLOSED_FRAMES = 0
    # cap = cv2.VideoCapture(video_path)
    # frames = []
    # labels = []
    # while True:
    #     ret, frame = cap.read()
    #     if not ret:
    #         break
    #     frame = cv2.resize(frame, size)
    #     frame = frame / 255.0  # Normalize
    #     frames.append(img_to_array(frame))
    #     labels.append(label)
    # cap.release()
    # return np.array(frames), np.array(labels)
    cap = cv2.VideoCapture(video_path)
    frames = []
    labels = []
    
    with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            if result.multi_face_landmarks:
                landmarks = result.multi_face_landmarks[0]
                left_eye = [(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in LEFT_EYE]
                right_eye = [(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in RIGHT_EYE]

                ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
                
                # Head pose landmarks (6 key points)
                # image_points = np.array([
                #     [landmarks.landmark[1].x * w, landmarks.landmark[1].y * h],     # Nose tip
                #     [landmarks.landmark[152].x * w, landmarks.landmark[152].y * h], # Chin
                #     [landmarks.landmark[263].x * w, landmarks.landmark[263].y * h], # Right eye corner
                #     [landmarks.landmark[33].x * w, landmarks.landmark[33].y * h],   # Left eye corner
                #     [landmarks.landmark[287].x * w, landmarks.landmark[287].y * h], # Right mouth corner
                #     [landmarks.landmark[57].x * w, landmarks.landmark[57].y * h],   # Left mouth corner
                # ], dtype="double")
        
                # focal_length = w
                # center = (w / 2, h / 2)
                # camera_matrix = np.array([
                #     [focal_length, 0, center[0]],
                #     [0, focal_length, center[1]],
                #     [0, 0, 1]
                # ], dtype="double")
        
                # dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
                # _, rot_vec, _ = cv2.solvePnP(FACE_3D_POINTS, image_points, camera_matrix, dist_coeffs)
                # rot_mat, _ = cv2.Rodrigues(rot_vec)
                # angles, _, _, _, _, _ = cv2.RQDecomp3x3(rot_mat)
                # pitch = angles[0]
                
                #label = 1 if ear < EAR_THRESHOLD else 0
                
                # Drowsiness logic with pitch
                if ear < EAR_THRESHOLD:  # looking down
                    CLOSED_FRAMES += 1
                    if CLOSED_FRAMES >= CONSEC_FRAMES:
                        label = 1  # drowsy
                    else:
                        label = 0  # still not enough frames to be sure
                else:
                    CLOSED_FRAMES = 0
                    label = 0  # awake


                resized = cv2.resize(frame, size) / 255.0
                frames.append(img_to_array(resized))
                labels.append(label)
                print(f"EAR: {ear:.2f}, ClosedFrames: {CLOSED_FRAMES}, Label: {label}")

    
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
y = to_categorical(np.array(y), num_classes=2)
 
# Save dataset for reuse
print("...........saving in progress..........")
np.save('X.npy', X)
np.save('y.npy', y)
print(".............saved..............")
print(f"Saved X shape: {X.shape}, y shape: {y.shape}")