import cv2
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array # type: ignore
import os

# Load model
model = load_model('model.h5')

# Create output directory if not exists
os.makedirs("classified_frames", exist_ok=True)

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(3, 320)
cap.set(4, 240)

frame_size = (64, 64)
THRESHOLD = 30
counter = 0
saved_frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        face = cv2.resize(frame, frame_size)
    except Exception as e:
        print(f"Resize failed: {e}")
        continue

    face = face / 255.0
    face = img_to_array(face)
    face = np.expand_dims(face, axis=0)

    pred = model.predict(face)[0]
    label = np.argmax(pred)

    if label == 1:
        counter += 1
        status = "DROWSY"
    else:
        counter = 0
        status = "AWAKE"

    # Save frame if label is predicted as drowsy or every N frames
    if status == "DROWSY" or saved_frame_count % 30 == 0:
        filename = f"classified_frames/frame_{saved_frame_count}_{status}.jpg"
        annotated_frame = frame.copy()
        cv2.putText(annotated_frame, f"Status: {status}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 255) if status == "DROWSY" else (0, 255, 0), 2)
        cv2.imwrite(filename, annotated_frame)
        saved_frame_count += 1

    cv2.putText(frame, f"Status: {status}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255) if status == "DROWSY" else (0, 255, 0), 2)

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()