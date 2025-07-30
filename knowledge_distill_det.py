from ultralytics import YOLO
from ultralytics.nn.attention.attention import add_attention
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.torch_utils import model_info


if __name__ == "__main__":
    # layers = ["6", "8", "13", "16", "19", "22"]
    layers = ["4", "6", "10", "16", "19", "22"]
    model_t = YOLO('runs/detect/yolo11/weights/best.pt')  # the teacher model
    model_s = YOLO("runs/detect/yolo11_prune_pruned/weights/best.pt")  # the student model
    model_s = add_attention(model_s) # Add attention to the student model
    
    # configure overrides
    overrides = {
        "model": "",
        "Distillation": model_t.model,
        "loss_type": "at",  #  {'cwd', 'mgd', 'at', 'skd', 'pkd'}
        "layers": layers,
        "epochs": 50,
        "imgsz": 640,
        "batch": 8,
        "device": 0,
        "lr0": 0.001,
        "amp": False,
        "sparse_training": False,
        "prune": False,
        "prune_load": False,
        "workers": 0,
        "data": "uno.yaml",
        "name": "yolo11_distill"
    }
    
    trainer = DetectionTrainer(overrides=overrides)
    trainer.model = model_s.model 
    
    print(f"Student model type: {type(trainer.model)}")
    print(f"Teacher model type: {type(model_t.model)}")
    
    model_info(trainer.model, verbose=True)
    trainer.train()
    