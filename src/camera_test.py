import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Opened:", cap.isOpened())

for i in range(10):

    ret, frame = cap.read()

    print(
        f"Frame {i}: ret={ret}",
        end=""
    )

    if ret:
        print(f", shape={frame.shape}")
    else:
        print()

cap.release()