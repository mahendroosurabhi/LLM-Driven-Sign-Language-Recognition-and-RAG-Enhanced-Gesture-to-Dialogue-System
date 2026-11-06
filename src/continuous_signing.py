import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
from collections import deque


# ============================================================
# 1. LOAD ASL MODEL
# ============================================================

MODEL_PATH = "models/ASL_final_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)

classes = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

print("ASL model loaded!")


# ============================================================
# 2. MEDIAPIPE HAND LANDMARKER
# ============================================================

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


# ============================================================
# 3. OPEN BRIO 100
# ============================================================

cap = cv2.VideoCapture(
    0,
    cv2.CAP_DSHOW
)


if not cap.isOpened():

    print("ERROR: Could not open Brio 100.")
    exit()


print("Webcam started.")
print()
print("Controls:")
print("Q = Quit")
print("C = Clear text")
print("S = Space")
print("P = Full stop")
print()


# ============================================================
# 4. SETTINGS
# ============================================================

CONFIDENCE_THRESHOLD = 0.75

# Number of frames used for smoothing
STABLE_FRAMES = 8

# Frames before another prediction can be accepted
COOLDOWN_FRAMES = 12


# ============================================================
# 5. VARIABLES
# ============================================================

prediction_history = deque(
    maxlen=STABLE_FRAMES
)

current_letter = ""

detected_text = ""

cooldown = 0

frame_timestamp = 0


# ============================================================
# 6. REPEATED LETTER CONTROL
# ============================================================

# Last letter that was accepted
last_committed_letter = None

# Allows the same letter to be recognized again
# after the user removes and shows the hand again.
letter_rearmed = True


# ============================================================
# 7. MAIN LOOP
# ============================================================

while True:

    # --------------------------------------------------------
    # Read webcam frame
    # --------------------------------------------------------

    ret, frame = cap.read()


    if not ret:

        print("ERROR: Could not read webcam frame.")
        break


    # Mirror webcam
    frame = cv2.flip(frame, 1)


    # --------------------------------------------------------
    # Convert BGR → RGB
    # --------------------------------------------------------

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # Create MediaPipe image
    # --------------------------------------------------------

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )


    frame_timestamp += 1


    # --------------------------------------------------------
    # MediaPipe hand detection
    # --------------------------------------------------------

    result = landmarker.detect_for_video(
        mp_image,
        frame_timestamp
    )


    # ========================================================
    # HAND DETECTED
    # ========================================================

    if result.hand_landmarks:

        hand_landmarks = result.hand_landmarks[0]

        h, w, _ = frame.shape


        # ----------------------------------------------------
        # Get hand coordinates
        # ----------------------------------------------------

        x_coords = [
            int(lm.x * w)
            for lm in hand_landmarks
        ]

        y_coords = [
            int(lm.y * h)
            for lm in hand_landmarks
        ]


        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        x_min = max(
            min(x_coords) - 30,
            0
        )

        y_min = max(
            min(y_coords) - 30,
            0
        )

        x_max = min(
            max(x_coords) + 30,
            w
        )

        y_max = min(
            max(y_coords) + 30,
            h
        )


        # ----------------------------------------------------
        # Draw bounding box
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (x_min, y_min),
            (x_max, y_max),
            (0, 255, 0),
            2
        )


        # ====================================================
        # CROP HAND
        # ====================================================

        hand = frame[
            y_min:y_max,
            x_min:x_max
        ]


        if hand.size != 0:

            # ------------------------------------------------
            # BGR → RGB
            # ------------------------------------------------

            hand_rgb = cv2.cvtColor(
                hand,
                cv2.COLOR_BGR2RGB
            )


            # ------------------------------------------------
            # Resize to model input
            # ------------------------------------------------

            hand_resized = cv2.resize(
                hand_rgb,
                (128, 128)
            )


            # ------------------------------------------------
            # Normalize
            # ------------------------------------------------

            input_image = (
                hand_resized.astype(
                    np.float32
                ) / 255.0
            )


            # Add batch dimension
            input_image = np.expand_dims(
                input_image,
                axis=0
            )


            # =================================================
            # MODEL PREDICTION
            # =================================================

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


            # =================================================
            # CONFIDENCE FILTER
            # =================================================

            if confidence >= CONFIDENCE_THRESHOLD:

                current_letter = predicted_letter

                prediction_history.append(
                    predicted_letter
                )

            else:

                current_letter = "?"

                prediction_history.clear()


            # =================================================
            # TEMPORAL SMOOTHING
            # =================================================

            if (
                len(prediction_history)
                == STABLE_FRAMES
            ):

                most_common = max(
                    set(prediction_history),
                    key=prediction_history.count
                )


                count = prediction_history.count(
                    most_common
                )
       

                # ------------------------------------------------
                # Stable prediction
                # ------------------------------------------------

                if count >= STABLE_FRAMES - 1:


                    if cooldown == 0:


                        # =================================================
                        # NEW LETTER
                        # =================================================

                        if (
                            most_common
                            != last_committed_letter
                        ):

                            detected_text += most_common

                            print(
                                "Detected:",
                                most_common
                            )

                            last_committed_letter = (
                                most_common
                            )

                            letter_rearmed = False

                            cooldown = (
                                COOLDOWN_FRAMES
                            )

                            prediction_history.clear()


                        # =================================================
                        # SAME LETTER
                        # =================================================

                        elif letter_rearmed:

                            detected_text += most_common

                            print(
                                "Detected:",
                                most_common
                            )

                            letter_rearmed = False

                            cooldown = (
                                COOLDOWN_FRAMES
                            )

                            prediction_history.clear()


    # ========================================================
    # NO HAND DETECTED
    # ========================================================

    else:

        current_letter = ""

        prediction_history.clear()


        # ----------------------------------------------------
        # Re-arm the same letter
        # ----------------------------------------------------

        letter_rearmed = True

        last_committed_letter = None


    # ========================================================
    # COOLDOWN
    # ========================================================

    if cooldown > 0:

        cooldown -= 1


    # ========================================================
    # DISPLAY CURRENT LETTER
    # ========================================================

    cv2.putText(
        frame,
        f"Current: {current_letter}",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )


    # ========================================================
    # DISPLAY TEXT
    # ========================================================

    cv2.putText(
        frame,
        f"Text: {detected_text}",
        (30, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    # ========================================================
    # DISPLAY CONTROLS
    # ========================================================

    cv2.putText(
        frame,
        "Q:Quit  C:Clear  S:Space  P:Period",
        (30, 140),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # ========================================================
    # SHOW WINDOW
    # ========================================================

    cv2.imshow(
        "Continuous ASL Recognition",
        frame
    )


    # ========================================================
    # KEYBOARD INPUT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    # --------------------------------------------------------
    # Q = QUIT
    # --------------------------------------------------------

    if key == ord("q"):

        break


    # --------------------------------------------------------
    # C = CLEAR
    # --------------------------------------------------------

    elif key == ord("c"):

        detected_text = ""

        prediction_history.clear()

        last_committed_letter = None

        letter_rearmed = True

        print("Text cleared.")


    # --------------------------------------------------------
    # S = SPACE
    # --------------------------------------------------------

    elif key == ord("s"):

        if (
            len(detected_text) > 0
            and not detected_text.endswith(" ")
        ):

            detected_text += " "

            print("SPACE added.")


    # --------------------------------------------------------
    # P = FULL STOP
    # --------------------------------------------------------

    elif key == ord("p"):

        # Remove trailing spaces first
        detected_text = detected_text.rstrip()

        # Don't add multiple periods
        if not detected_text.endswith("."):

            detected_text += "."

            print("FULL STOP added.")


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print()
print("Webcam closed.")
print("Final text:", detected_text)