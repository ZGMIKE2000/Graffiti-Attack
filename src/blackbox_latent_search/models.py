import sys
import torch
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)

def load_stylegan3_generator(snapshot_path: str, device: str = "cpu") -> torch.nn.Module:
    """
    Load a StyleGAN3 generator from a snapshot file.

    Args:
        snapshot_path (str): Path to the .pkl snapshot file.
        device (str): 'cpu' or 'cuda'.

    Returns:
        torch.nn.Module: The loaded generator model.
    """
    logging.info(f"Loading StyleGAN3 Generator from {snapshot_path}...")
    try:
        import legacy
        import dnnlib
        with dnnlib.util.open_url(snapshot_path) as f:
            data = legacy.load_network_pkl(f)
            G = data['G_ema'].to(device)
        logging.info("StyleGAN3 Generator loaded successfully.")
        return G
    except FileNotFoundError:
        logging.error(f"Snapshot file not found: {snapshot_path}")
        sys.exit(1)
    except ImportError as e:
        logging.error(f"Required module missing: {e}. Ensure 'legacy' and 'dnnlib' are installed.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error while loading StyleGAN3 generator: {e}")
        sys.exit(1)

def load_yolo_model(model_path: str, device: str = "cpu"):
    """
    Load a YOLOv8 model from a checkpoint file.

    Args:
        model_path (str): Path to the YOLO .pt file.
        device (str): 'cpu' or 'cuda'.

    Returns:
        YOLO: The loaded YOLO model.
    """
    logging.info(f"Loading YOLO model from {model_path}...")
    try:
        from ultralytics import YOLO
        model = YOLO(model_path).to(device)
        logging.info("YOLO model loaded successfully.")
        return model
    except FileNotFoundError:
        logging.error(f"Model file not found: {model_path}")
        sys.exit(1)
    except ImportError:
        logging.error("Ultralytics YOLO not installed. Please run 'pip install ultralytics'.")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unexpected error while loading YOLO model: {e}")
        sys.exit(1)