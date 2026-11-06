import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np

# -----------------------------------
# Load ASL model
# -----------------------------------

MODEL_PATH = "models/ASL_final_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)

classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

print("ASL model loaded!")


# -----------------------------------
# MediaPipe Hand Landmarker
# -----------------------------------

MODEL_HAND = "models/mediapipe/hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_HAND
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = HandLandmarker.create_from_options(options)


# -----------------------------------
# Open webcam
# -----------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Webcam started.")
print("Press Q to quit.")


frame_timestamp = 0


while True:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read webcam frame.")
        break

    # Mirror webcam
    frame = cv2.flip(frame, 1)

    # OpenCV BGR → RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # MediaPipe image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    # Detect hand
    frame_timestamp += 1

    result = landmarker.detect_for_video(
        mp_image,
        frame_timestamp
    )

    # -----------------------------------
    # If hand detected
    # -----------------------------------

    if result.hand_landmarks:

        hand_landmarks = result.hand_landmarks[0]

        h, w, _ = frame.shape

        # Get landmark coordinates
        x_coords = [
            int(lm.x * w)
            for lm in hand_landmarks
        ]

        y_coords = [
            int(lm.y * h)
            for lm in hand_landmarks
        ]

        # Bounding box with padding
        x_min = max(min(x_coords) - 30, 0)
        y_min = max(min(y_coords) - 30, 0)

        x_max = min(max(x_coords) + 30, w)
        y_max = min(max(y_coords) + 30, h)

        # Draw bounding box
        cv2.rectangle(
            frame,
            (x_min, y_min),
            (x_max, y_max),
            (0, 255, 0),
            2
        )

        # -----------------------------------
        # Crop hand
        # -----------------------------------

        hand = frame[
            y_min:y_max,
            x_min:x_max
        ]

        if hand.size != 0:

            # BGR → RGB
            hand_rgb = cv2.cvtColor(
                hand,
                cv2.COLOR_BGR2RGB
            )

            # Resize
            hand_resized = cv2.resize(
                hand_rgb,
                (128, 128)
            )

            # Normalize
            input_image = (
                hand_resized.astype(np.float32)
                / 255.0
            )

            # Add batch dimension
            input_image = np.expand_dims(
                input_image,
                axis=0
            )

            # -----------------------------------
            # ASL prediction
            # -----------------------------------

            prediction = model.predict(
                input_image,
                verbose=0
            )

            predicted_index = np.argmax(
                prediction[0]
            )

            predicted_letter = classes[
                predicted_index
            ]

            confidence = prediction[0][
                predicted_index
            ]

            # Display prediction
            text = (
                f"{predicted_letter} "
                f"({confidence * 100:.1f}%)"
            )

            cv2.putText(
                frame,
                text,
                (x_min, max(y_min - 10, 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    # -----------------------------------
    # Show webcam
    # -----------------------------------

    cv2.imshow(
        "ASL Recognition Test",
        frame
    )

    # Q = quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# -----------------------------------
# Cleanup
# -----------------------------------

cap.release()
landmarker.close()
cv2.destroyAllWindows()

print("Webcam closed.")