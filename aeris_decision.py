"""
AERIS Decision Engine
=====================
Core module of the AERIS (Active Evidence Retrieval and Intelligent
Surveillance) system. Implements confidence-based multi-camera decision
logic with zone-restricted person detection.

Pipeline:
    1. Run YOLOv11 detection on Camera 1 within the calibrated zone.
    2. If detection confidence >= threshold → accept (skip Camera 2).
    3. If confidence < threshold → query Camera 2 for additional evidence.
    4. Fuse evidence from both cameras and render the final decision.

Author : Shrey Kataria
License: MIT
"""

from ultralytics import YOLO
import cv2

# =================================
# LOAD MODEL
# =================================

model = YOLO("yolo11n.pt")


# =================================
# OPEN CAMERAS
# =================================

camera1 = cv2.VideoCapture("cameras/camera1.mp4")
camera2 = cv2.VideoCapture("cameras/camera2.mp4")


# =================================
# SETTINGS
# =================================

WIDTH = 640
HEIGHT = 360

CONFIDENCE_THRESHOLD = 0.80


# =================================
# CAMERA 1 ZONE
# =================================

ZONE1_X1 = 369
ZONE1_Y1 = 34
ZONE1_X2 = 467
ZONE1_Y2 = 184


# =================================
# CAMERA 2 ZONE
# =================================

ZONE2_X1 = 387
ZONE2_Y1 = 8
ZONE2_X2 = 532
ZONE2_Y2 = 163


# =================================
# 30 SECOND TEST
# =================================

fps = camera1.get(cv2.CAP_PROP_FPS)

if fps == 0:
    fps = 30

max_frames = int(fps * 30)

frame_number = 0


# =================================
# WINDOWS
# =================================

cv2.namedWindow(
    "AERIS - Camera 1",
    cv2.WINDOW_NORMAL
)

cv2.namedWindow(
    "AERIS - Camera 2",
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    "AERIS - Camera 1",
    WIDTH,
    HEIGHT
)

cv2.resizeWindow(
    "AERIS - Camera 2",
    WIDTH,
    HEIGHT
)

cv2.moveWindow(
    "AERIS - Camera 1",
    20,
    50
)

cv2.moveWindow(
    "AERIS - Camera 2",
    680,
    50
)


# =================================
# MAIN LOOP
# =================================

while frame_number < max_frames:

    # ---------------------------------
    # READ FRAMES
    # ---------------------------------

    success1, frame1 = camera1.read()
    success2, frame2 = camera2.read()

    if not success1 or not success2:
        break

    frame_number = frame_number + 1


    # ---------------------------------
    # RESIZE
    # ---------------------------------

    frame1 = cv2.resize(
        frame1,
        (WIDTH, HEIGHT)
    )

    frame2 = cv2.resize(
        frame2,
        (WIDTH, HEIGHT)
    )


    # =================================
    # CAMERA 1
    # =================================

    result1 = model(
        frame1,
        verbose=False
    )[0]

    confidence1 = 0

    person_found_camera1 = False


    # ---------------------------------
    # DRAW CAMERA 1 ZONE
    # ---------------------------------

    cv2.rectangle(
        frame1,
        (ZONE1_X1, ZONE1_Y1),
        (ZONE1_X2, ZONE1_Y2),
        (0, 0, 255),
        2
    )

    cv2.putText(
        frame1,
        "AERIS ANALYSIS ZONE",
        (ZONE1_X1, ZONE1_Y1 + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 255),
        1
    )


    # ---------------------------------
    # CAMERA 1 DETECTIONS
    # ---------------------------------

    for i in range(len(result1.boxes)):

        class_id = int(
            result1.boxes.cls[i]
        )

        confidence = float(
            result1.boxes.conf[i]
        )

        x1 = int(
            result1.boxes.xyxy[i][0]
        )

        y1 = int(
            result1.boxes.xyxy[i][1]
        )

        x2 = int(
            result1.boxes.xyxy[i][2]
        )

        y2 = int(
            result1.boxes.xyxy[i][3]
        )


        # Person
        if class_id == 0:

            center_x = int(
                (x1 + x2) / 2
            )

            center_y = int(
                (y1 + y2) / 2
            )


            # Check Camera 1 zone
            inside_zone1 = (
                ZONE1_X1 <= center_x <= ZONE1_X2
                and
                ZONE1_Y1 <= center_y <= ZONE1_Y2
            )


            if inside_zone1:

                person_found_camera1 = True

                if confidence > confidence1:

                    confidence1 = confidence


                # RED = relevant person
                box_color = (
                    0,
                    0,
                    255
                )


                # Draw box
                cv2.rectangle(
                    frame1,
                    (x1, y1),
                    (x2, y2),
                    box_color,
                    2
                )


                # Confidence
                cv2.putText(
                    frame1,
                    "Person: "
                    + str(round(confidence, 2)),
                    (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    box_color,
                    1
                )


                # Center
                cv2.circle(
                    frame1,
                    (center_x, center_y),
                    4,
                    box_color,
                    -1
                )


    # =================================
    # AERIS CAMERA 1 DECISION
    # =================================

    if not person_found_camera1:

        decision = "NO TARGET IN ZONE"

        request_camera2 = False

        final_confidence = 0

    elif confidence1 >= CONFIDENCE_THRESHOLD:

        decision = "CAMERA 1: SUFFICIENT"

        request_camera2 = False

        final_confidence = confidence1

    else:

        decision = "CAMERA 1: INSUFFICIENT"

        request_camera2 = True

        final_confidence = confidence1


    # =================================
    # CAMERA 2
    # =================================

    confidence2 = 0

    person_found_camera2 = False


    # ---------------------------------
    # DRAW CAMERA 2 ZONE
    # ---------------------------------

    cv2.rectangle(
        frame2,
        (ZONE2_X1, ZONE2_Y1),
        (ZONE2_X2, ZONE2_Y2),
        (0, 0, 255),
        2
    )

    cv2.putText(
        frame2,
        "AERIS ANALYSIS ZONE",
        (ZONE2_X1, ZONE2_Y1 + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 0, 255),
        1
    )


    # ---------------------------------
    # ONLY PROCESS CAMERA 2
    # IF NEEDED
    # ---------------------------------

    if request_camera2:

        result2 = model(
            frame2,
            verbose=False
        )[0]


        # ---------------------------------
        # CAMERA 2 DETECTIONS
        # ---------------------------------

        for i in range(len(result2.boxes)):

            class_id = int(
                result2.boxes.cls[i]
            )

            confidence = float(
                result2.boxes.conf[i]
            )

            x1 = int(
                result2.boxes.xyxy[i][0]
            )

            y1 = int(
                result2.boxes.xyxy[i][1]
            )

            x2 = int(
                result2.boxes.xyxy[i][2]
            )

            y2 = int(
                result2.boxes.xyxy[i][3]
            )


            # Person
            if class_id == 0:

                center_x = int(
                    (x1 + x2) / 2
                )

                center_y = int(
                    (y1 + y2) / 2
                )


                # Check Camera 2 zone
                inside_zone2 = (
                    ZONE2_X1 <= center_x <= ZONE2_X2
                    and
                    ZONE2_Y1 <= center_y <= ZONE2_Y2
                )


                if inside_zone2:

                    person_found_camera2 = True

                    if confidence > confidence2:

                        confidence2 = confidence


                    # RED = relevant person
                    box_color = (
                        0,
                        0,
                        255
                    )


                    # Draw box
                    cv2.rectangle(
                        frame2,
                        (x1, y1),
                        (x2, y2),
                        box_color,
                        2
                    )


                    # Confidence
                    cv2.putText(
                        frame2,
                        "Person: "
                        + str(round(confidence, 2)),
                        (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        box_color,
                        1
                    )


                    # Center
                    cv2.circle(
                        frame2,
                        (center_x, center_y),
                        4,
                        box_color,
                        -1
                    )


        # ---------------------------------
        # FUSE CAMERA 2 EVIDENCE
        # ---------------------------------

        if confidence2 > final_confidence:

            final_confidence = confidence2


    # =================================
    # FINAL DECISION
    # =================================

    if final_confidence >= CONFIDENCE_THRESHOLD:

        final_status = (
            "FINAL: SUFFICIENT EVIDENCE"
        )

    else:

        final_status = (
            "FINAL: INSUFFICIENT EVIDENCE"
        )


    # =================================
    # CAMERA 1 TEXT
    # =================================

    cv2.putText(
        frame1,
        "AERIS - CAMERA 1",
        (20, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame1,
        "Zone Confidence: "
        + str(round(confidence1, 2)),
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    cv2.putText(
        frame1,
        decision,
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        2
    )


    if request_camera2:

        cv2.putText(
            frame1,
            "REQUESTING CAMERA 2",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2
        )

    else:

        cv2.putText(
            frame1,
            "CAMERA 2 NOT NEEDED",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )


    # =================================
    # CAMERA 2 TEXT
    # =================================

    cv2.putText(
        frame2,
        "AERIS - CAMERA 2",
        (20, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2
    )


    if request_camera2:

        cv2.putText(
            frame2,
            "ADDITIONAL EVIDENCE",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame2,
            "Zone Confidence: "
            + str(round(confidence2, 2)),
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

    else:

        cv2.putText(
            frame2,
            "STANDBY",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )


    # =================================
    # FINAL STATUS
    # =================================

    cv2.putText(
        frame1,
        final_status,
        (20, 335),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2
    )


    # =================================
    # SHOW BOTH CAMERAS
    # =================================

    cv2.imshow(
        "AERIS - Camera 1",
        frame1
    )

    cv2.imshow(
        "AERIS - Camera 2",
        frame2
    )


    # =================================
    # PRESS Q TO STOP
    # =================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# =================================
# CLEANUP
# =================================

camera1.release()
camera2.release()

cv2.destroyAllWindows()

print()
print("================================")
print("AERIS TEST COMPLETED")
print("================================")