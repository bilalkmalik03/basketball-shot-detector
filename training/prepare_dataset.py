# training/prepare_dataset.py
import os
import cv2
from roboflow import Roboflow

# Use Roboflow for easy dataset management
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace().project("basketball-detection")
dataset = project.version(1).download("yolov8")

# Or create your own dataset
def create_dataset(video_dir, output_dir):
    """Extract frames and prepare for labeling"""
    os.makedirs(output_dir, exist_ok=True)
    
    for video_file in os.listdir(video_dir):
        cap = cv2.VideoCapture(os.path.join(video_dir, video_file))
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % 30 == 0:  # Every second at 30fps
                output_path = f"{output_dir}/frame_{video_file}_{frame_count}.jpg"
                cv2.imwrite(output_path, frame)
            
            frame_count += 1
        
        cap.release()