# 🦾 YOLOv11 Pruning & Distillation

A PyTorch-based pipeline for **training**, **static pruning**, and **knowledge distillation** of the YOLOv11 object detection model.

This project enables significant **model compression** by pruning redundant structures and transferring knowledge from a teacher model to a student model — all while maintaining strong detection accuracy.

> 📌 **Current Ultralytics Version**: `8.3.160`

------

## 📦 Features

- ✅ YOLOv11 base model training
- ✂️ Structured static pruning using `torch-pruning`
- 📚 Multiple distillation strategies: `CWD`, `MGD`, `AT`, `SKD`, `PKD`
- 🧠 Student model can optionally be enhanced with attention modules
- 🔌 ONNX model export & deployment
- 📈 FLOPs/Params calculation using `thop`

------

## 🔧 Installation

```sh
git clone https://github.com/your-repo/yolov11_prune_distillation.git
cd yolov11_prune_distillation

# Install torch-pruning
pip install torch-pruning

# Install other dependencies
pip install -r requirements.txt
```

------

## 🚂 Workflow Overview

### 1. 🔁 Training YOLOv11

```python
from ultralytics import YOLO

model = YOLO('yolo11.yaml')  # or use a pre-trained weight path
model.train(data='uno.yaml', epochs=100, imgsz=640, batch=8, device="0", name='yolo11')
```

------

### 2. ✂️ Static Pruning

Supports both **quick pruning** and **multi-phase sparse-prune training**.

```python
from ultralytics import YOLO

model = YOLO('runs/detect/yolo11/weights/best.pt')

def prunetrain(train_epochs, prune_epochs=0, quick_pruning=True, prune_ratio=0.5, 
               prune_iterative_steps=1, data='coco.yaml', name='yolo11', imgsz=640, 
               batch=8, device=[0], sparse_training=False):
    if not quick_pruning:
        print("Phase 1: Pre-training...")
        model.train(data=data, epochs=train_epochs, prune=False, sparse_training=sparse_training)
        
        print("Phase 2: Pruning...")
        best_weights = f"runs/detect/{name}_phase1/weights/best.pt"
        pruned_model = YOLO(best_weights)
        return pruned_model.train(data=data, epochs=prune_epochs, prune=True, prune_ratio=prune_ratio)
    else:
        return model.train(data=data, epochs=train_epochs, prune=True, prune_ratio=prune_ratio)

if __name__ == '__main__':
    prunetrain(
        train_epochs=10, 
        prune_epochs=20, 
        quick_pruning=False,
        prune_ratio=0.5,
        prune_iterative_steps=1,
        data='uno.yaml',
        batch=8,
        imgsz=640,
        device=[0],
        name='yolo11_prune',
        sparse_training=True
    )
```

------

### 3. 📚 Knowledge Distillation

Supports multiple loss strategies:

- `CWD`: Channel-Wise Distillation
- `MGD`: Masked Generative Distillation
- `AT`: Attention Transfer
- `SKD`: Spatial Knowledge Distillation
- `PKD`: Pearson Correlation-based Distillation

```python
from ultralytics import YOLO
from ultralytics.nn.attention.attention import add_attention
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils.torch_utils import model_info

if __name__ == "__main__":
    layers = ["4", "6", "10", "16", "19", "22"]

    model_t = YOLO('runs/detect/yolo11/weights/best.pt')           # Teacher
    model_s = YOLO('runs/detect/yolo11_prune_pruned/weights/best.pt')  # Student

    model_s = add_attention(model_s)  # Optional: Inject attention for better distillation

    overrides = {
        "model": model_s.ckpt_path,
        "Distillation": model_t.model,
        "loss_type": "at",  # 'cwd', 'mgd', 'at', 'skd', 'pkd'
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
    model_info(trainer.model, verbose=True)
    trainer.train()
```

------

## 📤 Model Export

Export your trained model to **ONNX** format for deployment:

```python
from ultralytics import YOLO

model = YOLO('runs/detect/yolo11_distill/weights/yolo11n.pt')
model.export(format='onnx')
```

------

## 🖼️ Inference Demo

Run inference on a single image:

```python
from ultralytics import YOLO

model = YOLO('runs/detect/yolo11/weights/best.pt')
model.predict('fruits.jpg', save=True, device=[0], line_width=2)
```

------

## 📊 Model Analysis

Install `thop` to calculate FLOPs and parameter counts:

```python
pip install thop
```

Use `calculate.py` in this repo to analyze your model’s complexity.

------

------

## 🤝 Contributing & Support

We welcome contributions! Feel free to:

- 📥 Submit a pull request for enhancements or bugfixes
- 📩 Open an issue for questions, suggestions, or bugs

------

## 🙏 Acknowledgements

- 💡 [**Ultralytics**](https://github.com/ultralytics/ultralytics): The core YOLO training framework.
- ✂️ [**Torch-Pruning**](https://github.com/VainF/Torch-Pruning): Channel-pruning library used in this repo.
- 📦 [YOLO-Pruning-RKNN](https://github.com/heyongxin233/YOLO-Pruning-RKNN): Related pruning reference.
- 🔁 [yolov11_prune_distillation_v2](https://github.com/garlic-byte/yolov11_prune_distillation_v2): Related distillation inspiration.

------

## 📬 Contact

For questions or collaboration opportunities, feel free to reach out via GitHub Issues or Discussions!
