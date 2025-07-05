# test_models.py
import sys
sys.path.append('/home/michele/hdd/stylegan3')  # Use the actual path to your stylegan3 repo
from models import load_stylegan3_generator, load_yolov8_model
import torch

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Update these paths to your actual model files
    stylegan3_path = "/path/to/stylegan3.pkl"
    yolov8_path = "/path/to/yolov8.pt"

    # Test StyleGAN3 loading
    try:
        G = load_stylegan3_generator(stylegan3_path, device)
        print("StyleGAN3 loaded and ready.")
    except Exception as e:
        print(f"StyleGAN3 loading failed: {e}")

    # Test YOLOv8 loading
    try:
        yolo = load_yolov8_model(yolov8_path, device)
        print("YOLOv8 loaded and ready.")
    except Exception as e:
        print(f"YOLOv8 loading failed: {e}")