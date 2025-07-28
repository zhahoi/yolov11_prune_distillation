from ultralytics import YOLO
from ultralytics.nn.attention.attention import ParallelPolarizedSelfAttention
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.torch_utils import model_info

def add_attention(model):
    at0 = model.model.model[4]
    n0 = at0.cv2.conv.out_channels
    at0.attention = ParallelPolarizedSelfAttention(n0)

    at1 = model.model.model[6]
    n1 = at1.cv2.conv.out_channels
    at1.attention = ParallelPolarizedSelfAttention(n1)

    at2 = model.model.model[8]
    n2 = at2.cv2.conv.out_channels
    at2.attention = ParallelPolarizedSelfAttention(n2)
    return model


if __name__ == "__main__":
    # layers = ["6", "8", "13", "16", "19", "22"]
    layers = ["4", "6", "10", "16", "19", "22"]
    model_t = YOLO('runs/detect/yolo11/weights/best.pt')  # the teacher model
    model_s = YOLO("runs/detect/yolo11_prune_pruned/weights/best.pt")  # the student model
    model_s = add_attention(model_s)
    
    # configure overrides
    overrides = {
        "model": "runs/detect/yolo11_prune_pruned/weights/best.pt",
        "Distillation": model_t.model,
        "loss_type": "mgd",
        "layers": ["4", "6", "10", "16", "19", "22"],
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
        "data": "data.yaml",
        "name": "yolo11_distill"
    }
    
    trainer = DetectionTrainer(overrides=overrides)
    trainer.model = model_s.model 
    model_info(trainer.model, verbose=True)
    trainer.train()
    