# Fit.Fest Hackathon Event Stream5

# Problem Statement: Drowsiness Detection: Use eye aspect ratio (EAR) from facial landmarks or a small CNN like MobileNetV2 to detect eye closure. Track eye state over time (e.g., using a sliding window) to identify closure >2 seconds.

**Requirements:**

- Python version 3.9.10 is required.
- Install dependencies listed in `requirements.txt` before running any scripts.
- Ensure you have the necessary dataset to process video frames.
- For best results, use a machine with a compatible GPU for model training and evaluation.Requires python version 3.9.10

1. Extract frames and labels using `extract_drowsiness_frames_and_labels.py`, and save them in `.npy` format.

2. Train the 3D Convolutional Neural Network (CNN) model with the extracted data sequences using `train_drowsiness_3dcnn.py`.

3. To evaluate the trained model, ensure the input is in `.npy` format. Use `extract_drowsiness_frames_and_labels.py` to extract validation data, then process it into temporal frame chunks with `prepare_drowsiness_sequences.py`.

4. Evaluate the model using `evaluate_drowsiness_model.py`, providing the processed validation data.

