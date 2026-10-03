import cv2
import numpy as np
import tensorflow as tf

# ==========================================
# 1. CONFIGURATION & MODEL LOADING
# ==========================================
MODEL_PATH = "ASL_final_model.keras"  # Ensure this file is in the same directory
IMG_SIZE = 128
CLASSES = [chr(i) for i in range(ord("A"), ord("Z") + 1)]  # ['A', 'B', 'C', ..., 'Z']

print("Loading ASL CNN model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# ==========================================
# 2. WEBCAM INITIALIZATION
# ==========================================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("\n--- TEST SCRIPT RUNNING ---")
print("Position your hand inside the GREEN bounding box.")
print("Press 'q' on your keyboard to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame from webcam.")
        break

    # Flip horizontally for natural mirror display
    frame = cv2.flip(frame, 1)
    view = frame.copy()
    h, w, _ = frame.shape

    # Define a 300x300 bounding box centered on the screen
    box_size = 300
    x1 = int((w - box_size) / 2)
    y1 = int((h - box_size) / 2)
    x2, y2 = x1 + box_size, y1 + box_size

    # Draw the green hand placement box
    cv2.rectangle(view, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Crop the hand ROI (Region of Interest)
    roi = frame[y1:y2, x1:x2]

    if roi.size != 0:
        # Preprocess ROI to match training pipeline (128x128 RGB, normalized to [0, 1])
        rgb_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        resized_roi = cv2.resize(rgb_roi, (IMG_SIZE, IMG_SIZE))
        normalized_roi = resized_roi.astype("float32") / 255.0
        input_tensor = np.expand_dims(normalized_roi, axis=0)  # Shape: (1, 128, 128, 3)

        # Run model inference
        predictions = model.predict(input_tensor, verbose=0)
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx])

        predicted_letter = CLASSES[class_idx]

        # Choose overlay color based on confidence (Green if >= 60%, Red otherwise)
        text_color = (0, 255, 0) if confidence >= 0.60 else (0, 0, 255)

        # Display Top Prediction & Confidence
        label = f"Prediction: {predicted_letter} ({confidence * 100:.1f}%)"
        cv2.putText(
            view,
            label,
            (x1, y1 - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            text_color,
            2
        )

        # Optional: Display top 3 probability scores on screen
        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_str = " | ".join(
            [f"{CLASSES[idx]}: {predictions[0][idx]*100:.0f}%" for idx in top_3_indices]
        )
        cv2.putText(
            view,
            f"Top 3: {top_3_str}",
            (20, h - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1
        )

    # Show live video window
    cv2.imshow("ASL Letter Model Tester", view)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()