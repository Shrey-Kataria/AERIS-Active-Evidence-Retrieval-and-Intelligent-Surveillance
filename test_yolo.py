from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model("test.jpeg", show=True)

input("Press Enter to close...")