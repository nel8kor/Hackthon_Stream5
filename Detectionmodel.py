#!/usr/bin/env py
# -*- coding: utf-8 -*-
# Drowsiness detection with head pose estimation using OpenCV and MediaPipe
import cv2 # type: ignore
import mediapipe as mp # type: ignore
import numpy as np # type: ignore
import time
import math
 
# FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
 
# Landmark sets
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
 
# EAR threshold
EAR_THRESHOLD = 0.25
FPS = 15
CONSEC_FRAMES = int(2 * FPS)
CLOSED_FRAMES = 0
 
# Head pose 3D model points
FACE_3D_POINTS = np.array([
    (0.0, 0.0, 0.0),         # Nose tip
    (0.0, -330.0, -65.0),    # Chin
    (-225.0, 170.0, -135.0), # Left eye corner
    (225.0, 170.0, -135.0),  # Right eye corner
    (-150.0, -150.0, -125.0),# Left mouth corner
    (150.0, -150.0, -125.0)  # Right mouth corner
])
 
def euclidean(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))
 
def eye_aspect_ratio(eye_pts):
    A = euclidean(eye_pts[1], eye_pts[5])
    B = euclidean(eye_pts[2], eye_pts[4])
    C = euclidean(eye_pts[0], eye_pts[3])
    return (A + B) / (2.0 * C)
 
# Start video
video_path = 'dataset\driver_looking_at_phone\3820195_dms_drowsy_1.mp4'  # Change to your actual file path
cap = cv2.VideoCapture(0)
cap.set(3, 320)
cap.set(4, 240)
 
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
 
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    status = "Detecting..."
 
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
 
        # EAR calculation
        left_eye = [(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in LEFT_EYE]
        right_eye = [(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in RIGHT_EYE]
        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
 
        # Head pose landmarks (6 key points)
        image_points = np.array([
            [landmarks.landmark[1].x * w, landmarks.landmark[1].y * h],     # Nose tip
            [landmarks.landmark[152].x * w, landmarks.landmark[152].y * h], # Chin
            [landmarks.landmark[263].x * w, landmarks.landmark[263].y * h], # Right eye corner
            [landmarks.landmark[33].x * w, landmarks.landmark[33].y * h],   # Left eye corner
            [landmarks.landmark[287].x * w, landmarks.landmark[287].y * h], # Right mouth corner
            [landmarks.landmark[57].x * w, landmarks.landmark[57].y * h],   # Left mouth corner
        ], dtype="double")
 
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")
 
        dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
        _, rot_vec, _ = cv2.solvePnP(FACE_3D_POINTS, image_points, camera_matrix, dist_coeffs)
        rot_mat, _ = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rot_mat)
        pitch = angles[0]
 
        # Drowsiness logic with pitch
        if ear < EAR_THRESHOLD and pitch < -10:  # looking down
            CLOSED_FRAMES += 1
            if CLOSED_FRAMES >= CONSEC_FRAMES:
                status = "DROWSY!"
        else:
            CLOSED_FRAMES = 0
            status = "Awake"
 
        # Overlay text
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
        cv2.putText(frame, f"Pitch: {pitch:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)
        cv2.putText(frame, f"Status: {status}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255) if "DROWSY" in status else (0, 255, 0), 2)
 
    cv2.imshow("Drowsiness + Head Pose", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
cap.release()
cv2.destroyAllWindows()