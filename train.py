# from prune import prunetrain
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO('yolo11.yaml')
    results = model.train(data='uno.yaml', epochs=100, imgsz=640, batch=8, device="0", name='yolo11', workers=0, prune=False)
    
    print("Loaded config:", model.args)
