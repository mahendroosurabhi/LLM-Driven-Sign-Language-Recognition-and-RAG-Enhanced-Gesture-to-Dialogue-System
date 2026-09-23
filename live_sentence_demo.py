import cv2
import numpy as np

from extracted_keypoints import download_models, make_landmarkers, frame_features, MAX_WIDTH
from sign_model import Recognizer

# --- tuning knobs -----------------------------------------------------
MIN_FRAMES = 10        # ignore segments shorter than this (accidental flicker)
MAX_FRAMES = 90        # force-cut a segment if it runs this long without a pause
PAUSE_FRAMES = 8        # consecutive no-hand frames that count as "sign finished"
CONF_THRESHOLD = 0.5    # only add a word if top-1 confidence is at least this
N_POSE_POINTS = 33      # indices 0:33 = pose, 33:54 = left hand, 54:75 = right hand


def put(view, text, y, color=(255, 255, 255), scale=0.7):
    cv2.putText(view, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(view, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def hands_present(vec):
    """vec: flat (225,) frame feature. True if either hand has any nonzero landmark."""
    p = vec.reshape(75, 3)
    return bool(p[N_POSE_POINTS:].any())


def wrap_sentence(words, max_chars=60):
    """Split the running sentence into lines so it doesn't run off screen."""
    lines, cur = [], ""
    for w in words:
        cand = (cur + " " + w).strip()
        if len(cand) > max_chars and cur:
            lines.append(cur)
            cur = w
        else:
            cur = cand
    if cur:
        lines.append(cur)
    return lines or [""]


def main():
    download_models()
    hand, pose = make_landmarkers()
    rec = Recognizer()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam (try index 1 instead of 0).")

    signing = False
    frames = []
    no_hand_run = 0
    sentence = []
    last_call = None  # (word, conf) shown briefly after each segment
    note = "Sign a word to begin. BACKSPACE=undo  c=clear  q=quit"

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        f = frame
        h, w = f.shape[:2]
        if w > MAX_WIDTH:
            f = cv2.resize(f, (MAX_WIDTH, int(h * MAX_WIDTH / w)))
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        vec, _ = frame_features(rgb, hand, pose)
        present = hands_present(vec)

        if not signing:
            if present:
                signing = True
                frames = [vec]
                no_hand_run = 0
        else:
            frames.append(vec)
            no_hand_run = 0 if present else no_hand_run + 1

            segment_done = no_hand_run >= PAUSE_FRAMES or len(frames) >= MAX_FRAMES
            if segment_done:
                # drop the trailing no-hand frames, they're just the pause
                useful = frames[:-no_hand_run] if no_hand_run else frames
                signing = False
                frames = []
                no_hand_run = 0

                if len(useful) >= MIN_FRAMES:
                    word, conf = rec.predict_frames(useful)[0]
                    last_call = (word, conf)
                    if conf >= CONF_THRESHOLD:
                        sentence.append(word)
                        note = "Sign the next word..."
                    else:
                        note = f"Low confidence ({conf:.2f}), not added"
                else:
                    note = "Too short, ignored"

        # --- draw ---
        view = cv2.flip(frame, 1)
        if signing:
            put(view, f"RECORDING... {len(frames)} frames", 30, (0, 0, 255))
        else:
            put(view, note, 30)

        if last_call:
            w_, c_ = last_call
            put(view, f"last: {w_} ({c_:.2f})", 60, (0, 255, 0))

        for i, line in enumerate(wrap_sentence(sentence)):
            put(view, line, view.shape[0] - 20 - 30 * (len(wrap_sentence(sentence)) - 1 - i),
                (255, 255, 0), 0.8)

        cv2.imshow("Sign sentence demo", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            sentence = []
            last_call = None
            note = "Cleared. Sign a word to begin."
        elif key in (8, 127):  # BACKSPACE (varies by platform)
            if sentence:
                sentence.pop()
                note = "Removed last word."

    cap.release()
    cv2.destroyAllWindows()
    print("Final sentence:", " ".join(sentence))


if __name__ == "__main__":
    main()