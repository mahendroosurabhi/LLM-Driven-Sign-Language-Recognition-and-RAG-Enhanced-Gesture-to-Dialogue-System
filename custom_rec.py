import argparse
import ast
import os
import time
from pathlib import Path

import cv2
import pandas as pd

# ---------------------------------------------------------------- settings --

ROOT = Path(r"D:\Compressed\archive_3")      # same ROOT as the notebook
CUSTOM_DIR = ROOT / "custom_videos"          # where your mp4s go
INDEX_CSV = Path("custom_index.csv")         # sits next to the notebook
SIGNER_ID = 999                              # you; 900+ won't clash with WLASL

COLUMNS = ["gloss", "video_id", "split", "signer_id", "bbox",
           "frame_start", "frame_end", "path", "source"]

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ------------------------------------------------------------------ camera --

def open_camera(index, width, height):
    backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
    cap = cv2.VideoCapture(index, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera {index}")
    for _ in range(10):          # let auto-exposure settle
        cap.read()
    return cap


def show(frame, lines, color=(255, 255, 255)):
    """Mirror for display only. The saved frames are never mirrored."""
    view = cv2.flip(frame, 1)
    for i, text in enumerate(lines):
        y = 30 + i * 32
        cv2.putText(view, text, (12, y), FONT, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(view, text, (12, y), FONT, 0.7, color, 2, cv2.LINE_AA)
    cv2.imshow("record", view)


# --------------------------------------------------------------- recording --

def record_take(cap, gloss, take_no, seconds, countdown):
    """Returns (frames, fps) or (None, None) if the user pressed q."""
    t0 = time.time()
    while True:
        left = countdown - (time.time() - t0)
        if left <= 0:
            break
        ok, frame = cap.read()
        if not ok:
            continue
        show(frame, [f'"{gloss}"  take {take_no}', f"starting in {int(left) + 1}"],
             (0, 200, 255))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return None, None

    frames, stamps = [], []
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        now = time.time()
        frames.append(frame.copy())
        stamps.append(now)
        show(frame, [f'"{gloss}"  RECORDING', f"{now - t0:.1f}s / {seconds:.1f}s"],
             (0, 0, 255))
        cv2.waitKey(1)
        if now - t0 >= seconds:
            break

    if len(frames) < 10:
        return [], None

    # measured fps, not the driver's claim -- webcams lie about this constantly
    fps = (len(frames) - 1) / (stamps[-1] - stamps[0])
    return frames, round(fps, 2)


def review(frames, fps, gloss, take_no):
    """Loop the take back. Returns True to keep, False to retake, None to quit."""
    delay = max(1, int(1000 / (fps or 30)))
    while True:
        for f in frames:
            show(f, [f'"{gloss}"  take {take_no}', "[k]eep   [r]etake   [q]uit"],
                 (0, 255, 0))
            key = cv2.waitKey(delay) & 0xFF
            if key in (ord("k"), ord("y"), 13, 32):
                return True
            if key == ord("r"):
                return False
            if key == ord("q"):
                return None


def write_clip(frames, fps, path):
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    return w, h


# ------------------------------------------------------------------- index --

def load_index():
    if INDEX_CSV.exists():
        df = pd.read_csv(INDEX_CSV)
        df["bbox"] = df["bbox"].apply(
            lambda b: ast.literal_eval(b) if isinstance(b, str) else b)
        return df
    return pd.DataFrame(columns=COLUMNS)


def append_row(row):
    df = load_index()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(INDEX_CSV, index=False)


def next_take_number(gloss):
    df = load_index()
    if len(df) == 0:
        return 1
    return int((df["gloss"] == gloss).sum()) + 1


# -------------------------------------------------------------------- main --

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--words", nargs="+", help="glosses to record")
    ap.add_argument("--words-file", help="text file, one gloss per line")
    ap.add_argument("--takes", type=int, default=10, help="takes per word")
    ap.add_argument("--seconds", type=float, default=2.5, help="clip length")
    ap.add_argument("--countdown", type=int, default=3)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()

    words = list(args.words or [])
    if args.words_file:
        words += [w.strip() for w in open(args.words_file) if w.strip()]
    if not words:
        ap.error("give me --words or --words-file")

    CUSTOM_DIR.mkdir(parents=True, exist_ok=True)
    cap = open_camera(args.camera, args.width, args.height)
    total = 0

    try:
        for gloss in words:
            take = next_take_number(gloss)
            target = take + args.takes - 1

            while take <= target:
                # idle: wait for SPACE
                skip = quit_all = False
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        continue
                    show(frame, [f'next: "{gloss}"  take {take} of {target}',
                                 "SPACE start   [n]ext word   [q]uit"])
                    key = cv2.waitKey(1) & 0xFF
                    if key == 32:
                        break
                    if key == ord("n"):
                        skip = True
                        break
                    if key == ord("q"):
                        quit_all = True
                        break
                if quit_all:
                    return
                if skip:
                    break

                frames, fps = record_take(cap, gloss, take, args.seconds, args.countdown)
                if frames is None:
                    return
                if not frames:
                    print("take too short, ignoring")
                    continue

                verdict = review(frames, fps, gloss, take)
                if verdict is None:
                    return
                if verdict is False:
                    continue

                video_id = f"c{SIGNER_ID}_{gloss}_{take:03d}"
                path = CUSTOM_DIR / f"{video_id}.mp4"
                w, h = write_clip(frames, fps, path)

                append_row({
                    "gloss": gloss,
                    "video_id": video_id,
                    "split": "train",          # reassigned later, see assign_splits
                    "signer_id": SIGNER_ID,
                    "bbox": [0, 0, w, h],      # whole frame; you are the only person in it
                    "frame_start": 1,
                    "frame_end": len(frames),  # real count, safe under either convention
                    "path": str(path),
                    "source": "custom",
                })
                total += 1
                print(f"saved {video_id}  {len(frames)} frames @ {fps} fps")
                take += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n{total} new clips -> {INDEX_CSV}")


# ------------------------------------------- called from the notebook later --

def assign_splits(df, val_per_word=2, test_per_word=2, seed=0):
    """Spread each word's takes across train/val/test instead of leaving all train."""
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    out = []
    for gloss, g in df.groupby("gloss", sort=False):
        g = g.copy()
        splits = (["val"] * val_per_word
                  + ["test"] * test_per_word
                  + ["train"] * max(0, len(g) - val_per_word - test_per_word))
        g["split"] = splits[:len(g)]
        out.append(g)
    return pd.concat(out, ignore_index=True)


if __name__ == "__main__":
    main()