import cv2
import numpy as np

from extracted_keypoints import download_models, make_landmarkers, frame_features, MAX_WIDTH
from sign_model import Recognizer

MIN_FRAMES = 10   # ignore recordings shorter than this (accidental key presses)


def put(view, text, y, color=(255, 255, 255)):
    cv2.putText(view, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(view, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def main():
    download_models()
    hand, pose = make_landmarkers()
    rec = Recognizer()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam (try index 1 instead of 0).")

    recording, frames, result, note = False, [], [], "Press SPACE, sign, press SPACE again"
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if recording:
            f = frame
            h, w = f.shape[:2]
            if w > MAX_WIDTH:                                   # same downscale as training
                f = cv2.resize(f, (MAX_WIDTH, int(h * MAX_WIDTH / w)))
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)            # NOT mirrored: same as dataset videos
            vec, _ = frame_features(rgb, hand, pose)
            frames.append(vec)

        view = cv2.flip(frame, 1)                               # mirror only what we show
        if recording:
            put(view, f"RECORDING... {len(frames)} frames", 30, (0, 0, 255))
        else:
            put(view, note, 30)
        for i, (word, p) in enumerate(result):
            put(view, f"{i + 1}. {word}  ({p:.2f})", 65 + 30 * i, (0, 255, 0))
        cv2.imshow("Sign demo", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == 32:                                           # SPACE
            if not recording:
                recording, frames, result = True, [], []
            else:
                recording = False
                if len(frames) >= MIN_FRAMES:
                    result = rec.predict_frames(frames)
                    print(len(frames), "frames ->", result)
                    note = "Press SPACE to sign again"
                else:
                    note = "Too short, try again (SPACE)"

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()