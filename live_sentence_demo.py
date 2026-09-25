import cv2
import numpy as np

from extracted_keypoints import download_models, make_landmarkers, frame_features, MAX_WIDTH
from sign_model import Recognizer
from chat_langchain import get_reply   # your LangChain file — must expose get_reply(list_of_words) -> str

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
    """Split a line of words so it doesn't run off screen."""
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


def wrap_text(text, max_chars=60):
    return wrap_sentence(text.split(), max_chars)


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
    last_call = None      # (word, conf) shown briefly after each segment
    last_reply = None     # LLM's reply text, shown until the next send
    note = "Sign a word to begin. BACKSPACE=undo  c=clear  s=send  q=quit"

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
                        note = "Sign the next word... (s=send)"
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

        y = 90
        for line in wrap_sentence(sentence):
            put(view, line, y, (255, 255, 0), 0.8)
            y += 30

        if last_reply:
            y += 10
            for line in wrap_text(f"reply: {last_reply}"):
                put(view, line, y, (255, 0, 255), 0.7)
                y += 28

        cv2.imshow("Sign sentence demo", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            sentence = []
            last_call = None
            last_reply = None
            note = "Cleared. Sign a word to begin."
        elif key in (8, 127):  # BACKSPACE (varies by platform)
            if sentence:
                sentence.pop()
                note = "Removed last word."
        elif key == ord("s"):
            if not sentence:
                note = "Nothing to send yet."
            else:
                note = "Sending..."
                cv2.imshow("Sign sentence demo", view)   # paint "Sending..." before we block
                cv2.waitKey(1)

                last_reply = get_reply(sentence)          # blocks here until the LLM responds
                sentence = []
                note = "Sign the next sentence... (s=send)"

    cap.release()
    cv2.destroyAllWindows()
    print("Final sentence:", " ".join(sentence))
    if last_reply:
        print("Last reply:", last_reply)


if __name__ == "__main__":
    main()