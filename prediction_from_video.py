import cv2
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore

# Load trained model
model = load_model("model.h5")

# Parameters
sequence_length = 10
frame_height, frame_width = 120, 160
label_names = {0: "Distracted", 1: "Drowsy"}

# Read video
cap = cv2.VideoCapture("dataset/val/Drowsy/360326202501_dms_drowsy_12.mp4")
frames = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=100)
    resized = cv2.resize(frame, (frame_width, frame_height))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    frames.append(normalized)

cap.release()
frames = np.array(frames)

# Generate 10-frame sequences
sequences = []
for i in range(len(frames) - sequence_length + 1):
    clip = frames[i:i + sequence_length]
    sequences.append(clip)

sequences = np.array(sequences)

# Predict
if len(sequences) == 0:
    print("❌ Not enough frames to form sequences.")
else:
    predictions = model.predict(sequences)
    
    # Prepare video writer (same frame size and FPS as original, adjust FPS if needed)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('all_clips_labeled.mp4', fourcc, 20.0, (frame_width, frame_height))

    # Write all frames in each sequence
    for i, (clip, pred) in enumerate(zip(sequences, predictions)):
        label = np.argmax(pred)
        confidence = np.max(pred)
        label_text = f"{label_names[label]} ({confidence*100:.1f}%)"
        print(f"Clip {i}: Predicted: {label_names[label]}, Confidence = {confidence:.2f}")
        for j, frame in enumerate(clip):
            frame_bgr = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            if j == sequence_length - 1:
                # Only annotate the last frame of the sequence
                cv2.putText(frame_bgr, label_text, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            out.write(frame_bgr)

    out.release()
    print("✅ Saved full labeled clips to 'all_clips_labeled.mp4'")


