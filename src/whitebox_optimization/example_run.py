import os
import sys
import torch
from torch.utils.data import DataLoader
from dataset import StopSignBBoxDataset
from graffiti_generator import GraffitiPatchGenerator
from ultralytics import YOLO

# Add project root to sys.path for imports
sys.path.append(os.path.abspath("../../"))
import dnnlib
import legacy
from graffiti_trainer import GraffitiTrainer

# --- Paths (update as needed) ---
image_dir = "../dataset/stop_signs/images"
label_dir = "../dataset/stop_signs/labels"
gan_ckpt = "../models/network-snapshot-001400.pkl"
yolo_ckpt = "../models/yolov8n_best.pt"

# --- Device setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.autograd.set_detect_anomaly(True)  # Enable anomaly detection for debugging

# --- Load dataset and dataloader ---
dataset = StopSignBBoxDataset(image_dir, label_dir, image_size=640)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# --- Load StyleGAN3 generator ---
with dnnlib.util.open_url(gan_ckpt) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device)
generator = GraffitiPatchGenerator(G, device=device)

# --- Load YOLOv8 model ---
yolo = YOLO(yolo_ckpt)
yolo.model.to(device)
# yolo.model.eval()  # Keep in train mode for adversarial optimization

# --- Prepare latent vector and optimizer ---
z_dim = getattr(G, "z_dim", 512)
z = torch.randn(1, z_dim, device=device, requires_grad=True)
optimizer = torch.optim.Adam([z], lr=0.05)

# --- Train using GraffitiTrainer ---
trainer = GraffitiTrainer(generator, yolo, dataloader, dataset, device=device)
trainer.train_latent(z, optimizer, num_steps=10, target_class=None, reduction="max")

# Notes:
# - This script performs adversarial optimization of a latent vector z to maximize YOLOv8 loss.
# - Results (composite images, logs, loss curves) are saved by GraffitiTrainer.
# - Adjust paths, learning rate, and num_steps as needed for your experiments.