import os
import sys
import torch

from torch.utils.data import DataLoader
from dataset import StopSignBBoxDataset
from graffiti_trainer import GraffitiTrainer
from graffiti_generator import GraffitiPatchGenerator
from ultralytics import YOLO

# added stylegan project root for its dnnlib and legacy modules, which loads the generator.
sys.path.append(os.path.abspath("../../"))
import dnnlib
import legacy

# --- Paths (update as needed) ---
image_dir = "../dataset/carla_graffiti_dataset.v1-carla.yolov8/train/images_ablation"
label_dir = "../dataset/carla_graffiti_dataset.v1-carla.yolov8/train/labels_ablation"
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
# z = torch.randn(1, z_dim, device=device, requires_grad=True)
# Example: Load from file
# z = torch.load("/home/michele/hdd/stylegan3_error/adversarial_finetuning/src/outputs_70_1g/latent_vectors/latent_step_101.pt").to(device)
# z.requires_grad = True
# optimizer = torch.optim.Adam([z], lr=0.01)


# --- Train using GraffitiTrainer ---
num_tests = 10

for n in range(num_tests):
    test_dir = f"test_80_carla_{n}"
    os.makedirs(test_dir, exist_ok=True)
    # Option 1: Random latent with different seeds
    z = torch.randn(1, z_dim, device=device, requires_grad=True)
    z.requires_grad = True

    optimizer = torch.optim.Adam([z], lr=0.05)

    # Option 2: Load from a list of initial latents
    # z = torch.load(f"init_latent_{n}.pt").to(device)
    # z.requires_grad = True

    # Set output directory for this run
    log_file = os.path.join(test_dir, "training.log")
    trainer = GraffitiTrainer(generator, yolo, dataloader, dataset, device=device, output_root=test_dir)
    trainer.train_latent(z, optimizer, num_steps=1000)
    # Move or copy results to test_dir as needed

# trainer = GraffitiTrainer(generator, yolo, dataloader, dataset, device=device)
# trainer.train_latent(z, optimizer, num_steps=1500)

# Notes:
# - This script performs adversarial optimization of a latent vector z to maximize YOLOv8 loss.
# - Results (composite images, logs, loss curves) are saved by GraffitiTrainer.
# - Adjust paths, learning rate, and num_steps as needed for your experiments.