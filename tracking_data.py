"""
AERIS Person Tracking Module
=============================
Uses YOLOv11 + ByteTrack to perform multi-object person tracking on a
single camera feed. Detects intrusion events when a tracked individual
enters the calibrated restricted zone and logs confidence-based status.

Author : Shrey Kataria
License: MIT
"""

from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.pt")

results = model.track(
    source="cameras/camera1.mp4",
    tracker="bytetrack.yaml",
    stream=True
)

# Restricted zone
ZONE_X1 = 950
ZONE_Y1 = 390
ZONE_X2 = 1250
ZONE_Y2 = 750

# People currently inside the zone
people_inside = set()

frame_number = 0

for result in results:

    frame_number = frame_number + 1

    frame = result.orig_img

    # Draw restricted zone
    cv2.rectangle(
        frame,
        (ZONE_X1, ZONE_Y1),
        (ZONE_X2, ZONE_Y2),
        (0, 0, 255),
        3
    )

    cv2.putText(
        frame,
        "RESTRICTED ZONE",
        (ZONE_X1, ZONE_Y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    # People inside zone in this frame
    current_people_inside = set()

    if result.boxes.id is not None:

        ids = result.boxes.id
        classes = result.boxes.cls
        confidences = result.boxes.conf
        boxes = result.boxes.xyxy

        for i in range(len(ids)):

            track_id = int(ids[i])
            class_id = int(classes[i])
            confidence = float(confidences[i])

            x1 = int(boxes[i][0])
            y1 = int(boxes[i][1])
            x2 = int(boxes[i][2])
            y2 = int(boxes[i][3])

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Only check people
            if class_id == 0:

                # Check whether person is inside zone
                inside_zone = (
                    ZONE_X1 <= center_x <= ZONE_X2
                    and
                    ZONE_Y1 <= center_y <= ZONE_Y2
                )

                if inside_zone:

                    current_people_inside.add(track_id)

                    # Prototype uncertainty
                    if confidence >= 0.80:
                        status = "HIGH CONFIDENCE"
                    else:
                        status = "LOW CONFIDENCE"

                    # Person has just entered the zone
                    if track_id not in people_inside:

                        print()
                        print("================================")
                        print("INTRUSION EVENT")
                        print("Person ID:", track_id)
                        print("Frame:", frame_number)
                        print("Confidence:", round(confidence, 2))
                        print("Status:", status)
                        print("================================")

                    # Display confidence status
                    if status == "HIGH CONFIDENCE":

                        cv2.putText(
                            frame,
                            "HIGH CONFIDENCE",
                            (50, 120),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 255, 0),
                            2
                        )

                    else:

                        cv2.putText(
                            frame,
                            "LOW CONFIDENCE - MORE EVIDENCE NEEDED",
                            (50, 120),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 0, 255),
                            2
                        )

                # Draw center point
                cv2.circle(
                    frame,
                    (center_x, center_y),
                    5,
                    (255, 0, 0),
                    -1
                )

                # Draw tracking ID
                cv2.putText(
                    frame,
                    "ID: " + str(track_id),
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

    # Update people inside the zone
    people_inside = current_people_inside

    # AERIS status
    cv2.putText(
        frame,
        "AERIS ACTIVE",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Show intrusion status
    if len(people_inside) > 0:

        cv2.putText(
            frame,
            "INTRUSION DETECTED",
            (50, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3
        )

    # Show video
    cv2.imshow(
        "AERIS - Active Surveillance",
        frame
    )

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()