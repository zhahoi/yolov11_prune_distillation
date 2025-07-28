# yolov11_prune_distillation
Pruning and Distillation of the YOLOv11 Model.

**This project can be used for training, static pruning, and knowledge distillation of the YOLOv11 network. It aims to reduce the number of model parameters while preserving inference accuracy as much as possible.**



## 🔧 Install Dependencies

```shell
pip install torch-pruning 
pip install -r requirements.txt
```



## 🚂 Training &Pruning&Knowledge Distillation

### 📊 YOLO11 Training Example

```python
### train.py
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO('yolo11.yaml')
    results = model.train(data='uno.yaml', epochs=100, imgsz=640, batch=8, device="0", name='yolo11', workers=0, prune=False)
```



### ✂️ YOLO11 Pruning Example

```python
### prune.py
from ultralytics import YOLO

# model = YOLO('yolo11.yaml')
model = YOLO('runs/detect/yolo11/weights/best.pt')

def prunetrain(train_epochs, prune_epochs=0, quick_pruning=True, prune_ratio=0.5, 
               prune_iterative_steps=1, data='coco.yaml', name='yolo11', imgsz=640, 
               batch=8, device=[0], sparse_training=False):
    if not quick_pruning:
        assert train_epochs > 0 and prune_epochs > 0, "Quick Pruning is not set. prune epochs must > 0."
        print("Phase 1: Normal training...")
        model.train(data=data, epochs=train_epochs, imgsz=imgsz, batch=batch, device=device, name=f"{name}_phase1", prune=False,
                    sparse_training=sparse_training)
        
        print("Phase 2: Pruning training...")
        best_weights = f"runs/detect/{name}_phase1/weights/best.pt"
        pruned_model = YOLO(best_weights)
        
        return pruned_model.train(data=data, epochs=prune_epochs, imgsz=imgsz, batch=batch, device=device, name=f"{name}_pruned", prune=True,
                           prune_ratio=prune_ratio, prune_iterative_steps=prune_iterative_steps)
    else:
        return model.train(data=data, epochs=train_epochs, imgsz=imgsz, batch=batch, device=device, 
                           name=name, prune=True, prune_ratio=prune_ratio, prune_iterative_steps=prune_iterative_steps)


if __name__ == '__main__':
    # Normal Pruning
    prunetrain(quick_pruning=False,       # Quick Pruning or not
            data='uno.yaml',          # Dataset config
            train_epochs=10,           # Epochs before pruning
            prune_epochs=20,           # Epochs after pruning 
            imgsz=640,                 # Input size
            batch=8,                   # Batch size
            device=[0],                # GPU devices
            name='yolo11_prune',             # Save name
            prune_ratio=0.5,           # Pruning Ratio (50%)
            prune_iterative_steps=1,   # Pruning Interative Steps
            sparse_training=True      # Experimental, Allow Sparse Training Before Pruning
    )
    # Quick Pruning (prune_epochs no need)
    # prunetrain(quick_pruning=True, data='coco.yaml', train_epochs=10, imgsz=640, batch=8, device=[0], name='yolo11', 
    #            prune_ratio=0.5, prune_iterative_steps=1)
```



### 🔎 YOLO11 Knowledge Distillation Example

```python
### knowledge_distillation.py
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
    model_s = add_attention(model_s) # Add attention to the student model
    
    # configure overrides
    overrides = {
        "model": "runs/detect/yolo11_prune_pruned/weights/best.pt",
        "Distillation": model_t.model,
        "loss_type": "mgd",
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
        "data": "data.yaml",
        "name": "yolo11_distill"
    }
    
    trainer = DetectionTrainer(overrides=overrides)
    trainer.model = model_s.model 
    model_info(trainer.model, verbose=True)
    trainer.train()
    
```



## 📤 Model Export

### Export to ONNX Format Example

```python
### export.py
from ultralytics import YOLO

model = YOLO('runs/detect/yolo11_distill/weights/yolo11n.pt')
print(model.model)
model.export(format='onnx')
```



## 🌞 Model Inference

### Image Inference Example

```python
### infer.py
from ultralytics import YOLO
model = YOLO('runs/detect/yolo11/weights/best.pt') # model = YOLO('prune.pt')
model.predict('fruits.jpg', save=True, device=[0], line_width=2)
```



## 🔢 Model Analysis

Use `thop` to easily calculate model parameters and FLOPs:

```bash
pip install thop
```

You can calculate model parameters and flops by using `calculate.py`



## 🤝 Contributing & Support

Feel free to submit issues or pull requests on GitHub for questions or suggestions!

## 📚 Acknowledgements

- Special thanks to [@VainF](https://github.com/VainF) for the contribution to the [Torch-Pruning](https://github.com/VainF/Torch-Pruning) project! This project relies on it for model pruning.
- Special thanks to [@Ultralytics](https://github.com/ultralytics) for the contribution to the [ultralytics](https://github.com/ultralytics/ultralytics) project! This project relies on it for the framework.
- [YOLO-Pruning-RKNN](https://github.com/heyongxin233/YOLO-Pruning-RKNN)
- [yolov11_prune_distillation_v2](https://github.com/garlic-byte/yolov11_prune_distillation_v2.git)
