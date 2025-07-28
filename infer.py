from ultralytics import YOLO
model = YOLO('runs/detect/yolo11/weights/best.pt') # model = YOLO('prune.pt')
model.predict('fruits.jpg', save=True, device=[0], line_width=2)

