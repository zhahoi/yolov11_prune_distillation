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