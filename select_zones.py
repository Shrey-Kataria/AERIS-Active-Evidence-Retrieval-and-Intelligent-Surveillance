"""
AERIS Zone Calibration Tool
============================
Interactive utility for defining restricted detection zones on each
camera feed. Uses OpenCV's ROI selector to let the user drag-select
the area of interest. Outputs pixel coordinates to be pasted into
aeris_decision.py.

Usage:
    python select_zones.py

Author : Shrey Kataria
License: MIT
"""

import cv2

camera1 = cv2.VideoCapture("cameras/camera1.mp4")
camera2 = cv2.VideoCapture("cameras/camera2.mp4")

success1, frame1 = camera1.read()
success2, frame2 = camera2.read()

camera1.release()
camera2.release()

if not success1 or not success2:
    print("Could not open videos.")
    exit()

# Resize
frame1 = cv2.resize(frame1, (640, 360))
frame2 = cv2.resize(frame2, (640, 360))

print()
print("========================================")
print("AERIS ZONE CALIBRATION")
print("========================================")
print()
print("First select the zone for CAMERA 1.")
print("Drag with your mouse.")
print("Press ENTER when finished.")
print()

# Select Camera 1 zone
zone1 = cv2.selectROI(
    "SELECT CAMERA 1 ZONE",
    frame1,
    False,
    False
)

cv2.destroyWindow("SELECT CAMERA 1 ZONE")

print()
print("Camera 1 selected:")
print(zone1)

print()
print("Now select the zone for CAMERA 2.")
print("Drag with your mouse.")
print("Press ENTER when finished.")
print()

# Select Camera 2 zone
zone2 = cv2.selectROI(
    "SELECT CAMERA 2 ZONE",
    frame2,
    False,
    False
)

cv2.destroyWindow("SELECT CAMERA 2 ZONE")

print()
print("Camera 2 selected:")
print(zone2)

# Convert OpenCV format
x1, y1, w1, h1 = zone1
x2, y2, w2, h2 = zone2

print()
print("========================================")
print("COPY THESE VALUES")
print("========================================")

print()
print("CAMERA 1:")
print("ZONE1_X1 =", x1)
print("ZONE1_Y1 =", y1)
print("ZONE1_X2 =", x1 + w1)
print("ZONE1_Y2 =", y1 + h1)

print()
print("CAMERA 2:")
print("ZONE2_X1 =", x2)
print("ZONE2_Y1 =", y2)
print("ZONE2_X2 =", x2 + w2)
print("ZONE2_Y2 =", y2 + h2)

print()
print("========================================")

input("Press Enter to close...")