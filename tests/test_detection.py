# tests/test_detection.py
import pytest
import cv2
import numpy as np
from src.pipeline import BasketballTrackingPipeline

def test_ball_detection():
    """Test ball detection accuracy"""
    # Create synthetic test image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Draw orange circle (basketball)
    cv2.circle(img, (320, 240), 30, (0, 165, 255), -1)
    
    pipeline = BasketballTrackingPipeline()
    detections = pipeline.detector.detect(img)
    
    assert len(detections['ball']) > 0
    assert detections['ball'][0].confidence > 0.5

def test_trajectory_physics():
    """Test physics validation"""
    from src.physics import PhysicsValidator
    
    # Create realistic trajectory
    t = np.linspace(0, 2, 50)  # 2 seconds
    x = 100 + 50 * t  # Horizontal motion
    y = 100 + 100 * t - 0.5 * 9.81 * 100 * t**2  # Parabolic motion
    
    trajectory = np.column_stack([x, y, t])
    
    validator = PhysicsValidator()
    result = validator.validate_trajectory(trajectory)
    
    assert result['valid'] == True
    assert 0.9 < result['gravity_ratio'] < 1.1