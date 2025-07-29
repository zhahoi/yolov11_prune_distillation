from ultralytics import YOLO

model = YOLO('runs/detect/yolo11_distill/weights/best.pt') # model = YOLO('prune.pt')
model.predict('uno3.png', save=True, device=[0], line_width=2)

