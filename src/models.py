import sys
import torch

def load_stylegan3_generator(snapshot_path, device="cpu"):
    """
    Load a StyleGAN3 generator from a snapshot file.

    Args:
        snapshot_path (str): Path to the .pkl snapshot file.
        device (str): 'cpu' or 'cuda'.

    Returns:
        torch.nn.Module: The loaded generator model.
    """
    print(f"Loading StyleGAN3 Generator from {snapshot_path}...")
    try:
        import legacy
        import dnnlib
        with dnnlib.util.open_url(snapshot_path) as f:
            data = legacy.load_network_pkl(f)
            G = data['G_ema'].to(device)
        print("Generator loaded successfully.")
        return G
    except Exception as e:
        print(f"Error loading StyleGAN3 generator: {e}")
        print("Ensure dnnlib and legacy are accessible and snapshot path is correct.")
        sys.exit(1)

def load_yolov8_model(model_path, device="cpu"):
    """
    Load a YOLOv8 model from a checkpoint file.

    Args:
        model_path (str): Path to the YOLOv8 .pt file.
        device (str): 'cpu' or 'cuda'.

    Returns:
        YOLO: The loaded YOLOv8 model.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Ultralytics YOLO not installed. Please run 'pip install ultralytics'.")
        sys.exit(1)
    print(f"Loading YOLOv8 model from {model_path}...")
    model = YOLO(model_path).to(device)
    print("YOLOv8 model loaded successfully.")
    return model