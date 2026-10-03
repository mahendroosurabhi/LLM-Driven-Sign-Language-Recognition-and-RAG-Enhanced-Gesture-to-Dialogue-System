import cv2
import numpy as np

from extracted_keypoints import download_models, make_landmarkers, frame_features, MAX_WIDTH
from sign_model import Recognizer
from letter_model import LetterRecognizer, make_hand_landmarker, crop_hand
from chat_langchain import get_reply

MIN_FRAMES = 10
MAX_FRAMES = 90
PAUSE_FRAMES = 8
CONF_THRESHOLD = 0.5
N_POSE_POINTS = 33
LETTER_CONF_THRESHOLD = 0.5


def put(view, text, y, color=(255, 255, 255), scale=0.7):
    cv2.putText(view, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(view, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def hands_present(vec):
    p = vec.reshape(75, 3)
    return bool(p[N_POSE_POINTS:].any())


def wrap_sentence(words, max_chars=60):
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
    letter_rec = LetterRecognizer()   # loads ASL_final_model.keras
    letter_landmarker = make_hand_landmarker()   # loads models/mediapipe/hand_landmarker.task
    letter_timestamp = 0

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam (try index 1 instead of 0).")

    mode = "word"   # "word" or "letter"

    signing = False
    frames = []
    no_hand_run = 0
    sentence = []
    last_call = None
    last_reply = None
    note = "WORD mode. l=letter mode  s=send  c=clear  q=quit"

    spell_buffer = ""
    last_letter = None   # (letter, conf) shown live in letter mode

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        f = frame
        h, w = f.shape[:2]
        if w > MAX_WIDTH:
            f = cv2.resize(f, (MAX_WIDTH, int(h * MAX_WIDTH / w)))
        f_mirror = cv2.flip(f, 1)   # what the user actually sees - letter mode uses THIS
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        vec, _ = frame_features(rgb, hand, pose)

        if mode == "word":
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
                    useful = frames[:-no_hand_run] if no_hand_run else frames
                    signing = False
                    frames = []
                    no_hand_run = 0

                    if len(useful) >= MIN_FRAMES:
                        word, conf = rec.predict_frames(useful)[0]
                        last_call = (word, conf)
                        if conf >= CONF_THRESHOLD:
                            if word.lower() == "backspace":
                                if sentence:
                                    removed = sentence.pop()
                                    note = f"Removed '{removed}'. Sign the next word..."
                                else:
                                    note = "Nothing to remove."
                            else:
                                sentence.append(word)
                                note = "Sign the next word... (s=send, l=letter mode)"
                        else:
                            note = f"Low confidence ({conf:.2f}), not added"
                    else:
                        note = "Too short, ignored"

        else:  # mode == "letter"
            letter_timestamp += 1
            crop, bbox = crop_hand(f_mirror, letter_landmarker, letter_timestamp)
            if crop is not None:
                letter, conf = letter_rec.predict_crop(crop)[0]
                last_letter = (letter, conf)
            else:
                last_letter = None
                bbox = None

        # --- draw ---
        view = f_mirror.copy()

        if mode == "word":
            if signing:
                put(view, f"RECORDING... {len(frames)} frames", 30, (0, 0, 255))
            else:
                put(view, note, 30)
            if last_call:
                w_, c_ = last_call
                put(view, f"last: {w_} ({c_:.2f})", 60, (0, 255, 0))
        else:
            put(view, "LETTER mode. SPACE=add  ENTER=commit word  l=word mode", 30)
            if last_letter:
                l_, c_ = last_letter
                color = (0, 255, 0) if c_ >= LETTER_CONF_THRESHOLD else (0, 165, 255)
                put(view, f"seeing: {l_} ({c_:.2f})", 60, color)
            put(view, f"spelling: {spell_buffer}", 90, (255, 255, 0), 0.9)
            if bbox:
                x_min, y_min, x_max, y_max = bbox
                cv2.rectangle(view, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        y = 120
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

        elif key == ord("l"):
            if mode == "word":
                mode = "letter"
                spell_buffer = ""
                signing, frames, no_hand_run = False, [], 0
                note = "Switched to LETTER mode."
            else:
                mode = "word"
                note = "Switched to WORD mode. (unsent letters in buffer were dropped)"
                spell_buffer = ""

        elif key == ord("c"):
            sentence = []
            last_call = None
            last_reply = None
            spell_buffer = ""
            note = "Cleared."

        elif key in (8, 127):  # BACKSPACE
            if mode == "letter" and spell_buffer:
                spell_buffer = spell_buffer[:-1]
            elif mode == "word" and sentence:
                sentence.pop()
                note = "Removed last word."

        elif key == 32 and mode == "letter":  # SPACE: commit current letter
            if last_letter and last_letter[1] >= LETTER_CONF_THRESHOLD:
                spell_buffer += last_letter[0]
            else:
                note = "No confident letter to add."

        elif key == 13:  # ENTER: finalize spelled word
            if mode == "letter" and spell_buffer:
                sentence.append(spell_buffer)
                spell_buffer = ""
                note = "Word added. Keep spelling, or l=word mode."

        elif key == ord("s") and mode == "word":
            if not sentence:
                note = "Nothing to send yet."
            else:
                note = "Sending..."
                cv2.imshow("Sign sentence demo", view)
                cv2.waitKey(1)
                last_reply = get_reply(sentence)
                sentence = []
                note = "Sign the next sentence... (s=send)"

    cap.release()
    cv2.destroyAllWindows()
    print("Final sentence:", " ".join(sentence))
    if last_reply:
        print("Last reply:", last_reply)


if __name__ == "__main__":
    main()