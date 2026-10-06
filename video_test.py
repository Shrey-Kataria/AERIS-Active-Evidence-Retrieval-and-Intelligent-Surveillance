from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.track(
    source="cameras/camera1.mp4",
    show=True,
    tracker="bytetrack.yaml"
)

input("Press Enter to close...")