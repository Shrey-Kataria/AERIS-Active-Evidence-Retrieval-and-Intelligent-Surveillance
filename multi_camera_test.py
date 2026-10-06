"""
Multi-Camera Baseline Test
============================
Runs both camera feeds side-by-side with YOLOv11 person detection on
Camera 1 for 30 seconds. Serves as a comparison baseline — processes
Camera 1 every frame and simply displays Camera 2 without conditional
querying (unlike the AERIS engine).

Author : Shrey Kataria
License: MIT
"""

from ultralytics import YOLO
import cv2
import time

model = YOLO("yolo11n.pt")

camera1 = cv2.VideoCapture("cameras/camera1.mp4")
camera2 = cv2.VideoCapture("cameras/camera2.mp4")

# Process only first 30 seconds
fps = camera1.get(cv2.CAP_PROP_FPS)

if fps == 0:
    fps = 30

max_frames = int(fps * 30)

frame_number = 0

# Window settings
cv2.namedWindow("Camera 1", cv2.WINDOW_NORMAL)
cv2.namedWindow("Camera 2", cv2.WINDOW_NORMAL)

cv2.resizeWindow("Camera 1", 640, 360)
cv2.resizeWindow("Camera 2", 640, 360)

cv2.moveWindow("Camera 1", 20, 50)
cv2.moveWindow("Camera 2", 680, 50)

while frame_number < max_frames:

    success1, frame1 = camera1.read()
    success2, frame2 = camera2.read()

    if not success1 or not success2:
        break

    frame_number = frame_number + 1

    # Resize
    frame1 = cv2.resize(frame1, (640, 360))
    frame2 = cv2.resize(frame2, (640, 360))

    # Run YOLO
    result1 = model(frame1, verbose=False)[0]

    # Highest person confidence
    confidence1 = 0

    for i in range(len(result1.boxes)):

        class_id = int(result1.boxes.cls[i])
        confidence = float(result1.boxes.conf[i])

        if class_id == 0 and confidence > confidence1:
            confidence1 = confidence

    # Status
    if confidence1 >= 0.80:
        status = "SUFFICIENT EVIDENCE"
    else:
        status = "INSUFFICIENT EVIDENCE"

    # Camera 1
    cv2.putText(
        frame1,
        "AERIS - CAMERA 1",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame1,
        "Person Confidence: " + str(round(confidence1, 2)),
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame1,
        status,
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2
    )

    # Camera 2
    cv2.putText(
        frame2,
        "AERIS - CAMERA 2",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    # Show
    cv2.imshow("Camera 1", frame1)
    cv2.imshow("Camera 2", frame2)

    # Press Q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera1.release()
camera2.release()

cv2.destroyAllWindows()

print("30-second test completed.")