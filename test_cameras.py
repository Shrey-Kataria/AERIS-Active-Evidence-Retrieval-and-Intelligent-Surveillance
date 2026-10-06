import cv2

camera1 = cv2.VideoCapture("cameras/camera1.mp4")
camera2 = cv2.VideoCapture("cameras/camera2.mp4")

if camera1.isOpened():
    print("Camera 1 video opened successfully")
else:
    print("Camera 1 video failed")

if camera2.isOpened():
    print("Camera 2 video opened successfully")
else:
    print("Camera 2 video failed")

camera1.release()
camera2.release()

input("Press Enter to close...")