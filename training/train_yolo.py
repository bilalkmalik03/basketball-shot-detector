# training/train_yolo.py
from ultralytics import YOLO
import os
import yaml
from pathlib import Path

def main():
    # Set up paths
    current_dir = Path(__file__).parent  # training folder
    root_dir = current_dir.parent  # basketball-shot-detector
    
    # Look for Roboflow dataset
    roboflow_path = root_dir / "roboflow_dataset" / "data.yaml"
    
    if not roboflow_path.exists():
        print(f"ERROR: Cannot find data.yaml at {roboflow_path}")
        print("\nPlease extract your Roboflow zip file:")
        print("  unzip your-roboflow-file.zip -d ../roboflow_dataset/")
        return
    
    # Fix paths in data.yaml
    with open(roboflow_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Update path to absolute
    data['path'] = str((root_dir / "roboflow_dataset").absolute())
    
    with open(roboflow_path, 'w') as f:
        yaml.dump(data, f)
    
    print(f"Training with dataset: {data['path']}")
    print(f"Classes: {data['names']}")
    
    # Train model
    model = YOLO("yolov8n.pt")
    
    model.train(
        data=str(roboflow_path),
        epochs=100,  # Increased epochs
        imgsz=640,
        batch=8,  # Reduced batch size for memory
        project=str(root_dir / "models"),
        name="basketball_hoop",
        patience=20,
        exist_ok=True
    )
    
    # Validate
    model.val()
    
    print(f"\n✅ Training complete!")
    print(f"Model saved at: {root_dir}/models/basketball_hoop/weights/best.pt")

if __name__ == "__main__":
    main()