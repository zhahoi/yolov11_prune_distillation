from ultralytics import YOLO

model = YOLO('runs/detect/yolo11_distill/weights/yolo11n.pt')
print(model.model)
model.export(format='onnx')





