import string

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

IMG_SIZE = 128
CLASSES = list(string.ascii_uppercase)   # A-Z, confirmed to match the notebook's CLASSES order


class LetterRecognizer:
    def __init__(self, model_path="ASL_final_model.keras"):
        self.model = tf.keras.models.load_model(model_path)

    def predict_crop(self, frame_bgr, k=3):
        """frame_bgr: a BGR image (ideally already cropped to just the hand)."""
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
        probs = self.model.predict(img[None], verbose=0)[0]
        top = probs.argsort()[::-1][:k]
        return [(CLASSES[i], round(float(probs[i]), 3)) for i in top]


def make_hand_landmarker(task_path="models/hand_landmarker_letter.task"):
    """Dedicated landmarker for letter-mode cropping - separate from whatever
    your GRU word-mode pipeline uses, since that one only exposes a bool."""
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=task_path),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return HandLandmarker.create_from_options(options)


def crop_hand(frame_bgr, landmarker, timestamp_ms, pad=30):
    """Runs the dedicated landmarker on frame_bgr and returns (crop, bbox) or
    (None, None) if no hand is detected. bbox = (x_min, y_min, x_max, y_max),
    handy for drawing a rectangle like the reference scripts do.
    No square-padding here on purpose - the reference script that worked
    well just uses the raw padded box, so matching that rather than guessing."""
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    if not result.hand_landmarks:
        return None, None

    lm = result.hand_landmarks[0]
    xs = [int(p.x * w) for p in lm]
    ys = [int(p.y * h) for p in lm]

    x_min = max(min(xs) - pad, 0)
    y_min = max(min(ys) - pad, 0)
    x_max = min(max(xs) + pad, w)
    y_max = min(max(ys) + pad, h)

    crop = frame_bgr[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        return None, None
    return crop, (x_min, y_min, x_max, y_max)