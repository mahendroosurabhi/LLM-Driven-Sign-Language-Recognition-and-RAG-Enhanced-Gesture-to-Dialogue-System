import os
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

N_FRAMES = 32          # frames kept per video
MAX_WIDTH = 640      
FEAT_DIM = (33 + 21 + 21) * 3   # 225

MODEL_DIR = "models"
HAND_MODEL = f"{MODEL_DIR}/hand_landmarker.task"
POSE_MODEL = f"{MODEL_DIR}/pose_landmarker_full.task"
MODEL_URLS = {
    HAND_MODEL: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    POSE_MODEL: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
}


def download_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    for path, url in MODEL_URLS.items():
        if not os.path.exists(path):
            print("downloading", path)
            urllib.request.urlretrieve(url, path)


def make_landmarkers():
    """Create the hand and pose detectors (IMAGE mode: each frame is independent)."""
    hand = vision.HandLandmarker.create_from_options(
        vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
        )
    )
    pose = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
        )
    )
    return hand, pose


def _to_arr(landmarks, aspect):
    return np.array([[p.x * aspect, p.y, p.z] for p in landmarks]).flatten()


def frame_features(rgb, hand, pose):
    """One RGB frame -> (225,) feature vector and a 'was a hand found' flag."""
    h, w = rgb.shape[:2]
    aspect = w / h
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

    pose_res = pose.detect(mp_img)
    hand_res = hand.detect(mp_img)

    pose_vec = np.zeros(33 * 3)
    if pose_res.pose_landmarks:
        pose_vec = _to_arr(pose_res.pose_landmarks[0], aspect)

    # slots: 0 = "Left" label, 1 = "Right" label. We feed frames unflipped in both the
    # dataset and the live webcam, so the label convention is consistent everywhere.
    slots = [np.zeros(21 * 3), np.zeros(21 * 3)]
    filled = [False, False]
    for lms, handed in zip(hand_res.hand_landmarks, hand_res.handedness):
        slot = 0 if handed[0].category_name == "Left" else 1
        if filled[slot]:                 # both hands got the same label -> use the free slot
            slot = 1 - slot
            if filled[slot]:
                continue
        slots[slot] = _to_arr(lms, aspect)
        filled[slot] = True

    vec = np.concatenate([pose_vec, slots[0], slots[1]])
    return vec, any(filled)


def sample_indices(start, end, n=N_FRAMES):
    """n evenly spaced frame indices between start and end (short clips repeat frames)."""
    return np.linspace(start, end, n).round().astype(int)


def extract_video(row, hand, pose):
    """row needs: path, frame_start, frame_end (WLASL convention: 1-indexed, -1 = to the end).
    Returns (seq of shape (N_FRAMES, 225), fraction of frames where a hand was found)."""
    cap = cv2.VideoCapture(row["path"])
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = max(int(row["frame_start"]) - 1, 0)
    end = total - 1 if int(row["frame_end"]) == -1 else min(int(row["frame_end"]) - 1, total - 1)
    if end <= start:                     # bad annotation -> use whole clip
        start, end = 0, total - 1

    idxs = sample_indices(start, end)
    wanted = set(idxs.tolist())

    feats, found, i = {}, {}, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in wanted:
            h, w = frame.shape[:2]
            if w > MAX_WIDTH:
                frame = cv2.resize(frame, (MAX_WIDTH, int(h * MAX_WIDTH / w)))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # OpenCV is BGR, MediaPipe wants RGB
            feats[i], found[i] = frame_features(rgb, hand, pose)
        i += 1
        if i > idxs.max():
            break
    cap.release()

    if not feats:
        return np.zeros((N_FRAMES, FEAT_DIM)), 0.0

    keys = np.array(sorted(feats))
    seq, hit = [], []
    for k in idxs:
        k = k if k in feats else keys[np.abs(keys - k).argmin()]   # nearest frame if one was unreadable
        seq.append(feats[k])
        hit.append(found[k])
    return np.stack(seq), float(np.mean(hit))


def debug_draw(row, hand, pose, n_show=6, out_path="debug.jpg"):
    """Draw detected points on a few frames and save one image, to eyeball detection quality."""
    cap = cv2.VideoCapture(row["path"])
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    picks = set(np.linspace(0, total - 1, n_show).round().astype(int).tolist())
    tiles, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in picks:
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (480, int(h * 480 / w)))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
            fh, fw = frame.shape[:2]
            for lms in pose.detect(mp_img).pose_landmarks:
                for p in lms:
                    cv2.circle(frame, (int(p.x * fw), int(p.y * fh)), 3, (0, 255, 0), -1)
            for lms in hand.detect(mp_img).hand_landmarks:
                for p in lms:
                    cv2.circle(frame, (int(p.x * fw), int(p.y * fh)), 3, (0, 0, 255), -1)
            tiles.append(frame)
        i += 1
    cap.release()
    if tiles:
        th = min(t.shape[0] for t in tiles)
        tiles = [t[:th] for t in tiles]
        cv2.imwrite(out_path, np.hstack(tiles))
    return out_path


def run_all(df, hand, pose, out_dir="data/keypoints"):
    """Extract every video in df, save <video_id>.npy, skip files already done (resumable).
    Returns df with a 'hand_rate' column."""
    os.makedirs(out_dir, exist_ok=True)
    rates = []
    for n, (_, row) in enumerate(df.iterrows()):
        f = f"{out_dir}/{row['video_id']}.npy"
        rate_f = f"{out_dir}/{row['video_id']}.rate.txt"
        if os.path.exists(f) and os.path.exists(rate_f):
            rates.append(float(open(rate_f).read()))
            continue
        seq, rate = extract_video(row, hand, pose)
        np.save(f, seq.astype(np.float32))
        open(rate_f, "w").write(str(rate))
        rates.append(rate)
        if n % 50 == 0:
            print(f"{n}/{len(df)} done")
    out = df.copy()
    out["hand_rate"] = rates
    return out