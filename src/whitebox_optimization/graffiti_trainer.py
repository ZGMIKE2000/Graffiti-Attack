import torch
import torch.nn.functional as F
from patch_overlay import overlay_patch_rgba
import matplotlib.pyplot as plt
from ultralytics.utils.loss import v8DetectionLoss
import itertools
import logging

class GraffitiTrainer:
    """
    Trainer for adversarial latent optimization using a StyleGAN3 generator and YOLOv8 detector.
    Applies a generated patch to images, computes detection loss, and optimizes the latent vector.
    """

    def __init__(self, generator, yolo, dataloader, dataset, device="cuda", log_file="training.log"):
        """
        Args:
            generator: GraffitiPatchGenerator instance (wraps StyleGAN3 generator)
            yolo: Ultralytics YOLO model (with .model attribute)
            dataloader: PyTorch DataLoader yielding batches with 'orig_img' and 'orig_bboxes'
            dataset: Dataset object (for saving composites)
            device: Device string, e.g. "cuda" or "cpu"
            log_file: Path to save training logs
        """
        self.generator = generator
        self.yolo = yolo
        self.dataloader = dataloader
        self.dataset = dataset
        self.device = device
        self.yolo_loss = v8DetectionLoss(self.yolo.model)  # Direct YOLOv8 loss

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.FileHandler(log_file, mode='w'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger("GraffitiTrainer")

    def train_latent(self, z, optimizer, num_steps=10, target_class=None, reduction="max"):
        """
        Optimize the latent vector z to maximize YOLOv8 detection loss on patched images.

        Args:
            z: Latent vector (torch.Tensor, requires_grad=True)
            optimizer: Optimizer for z
            num_steps: Number of optimization steps
            target_class: (Unused) Optionally focus on a specific class
            reduction: (Unused) Loss reduction method
        """
        loss_history = []
        dataloader_iter = itertools.cycle(self.dataloader)  # Infinite iterator over batches

        for step in range(num_steps):
            batch = next(dataloader_iter)

            # --- Extract bounding boxes and original image ---
            orig_bboxes = batch.get("orig_bboxes", [])
            orig_img_tensor = batch["orig_img"].to(self.device)

            # --- Generate patch and mask ---
            patch = self.generator.generate(z)
            mask, _ = self.generator.auto_quantile_mask(patch)
            patch_rgba = torch.cat([patch, mask], dim=1)

            # --- Overlay patch on all bboxes in the image ---
            composite_tensor = orig_img_tensor.clone()
            for bbox in orig_bboxes:
                _, x1, y1, x2, y2 = bbox
                bbox_h, bbox_w = y2 - y1, x2 - x1
                patch_resized = F.interpolate(patch_rgba, size=(bbox_h, bbox_w), mode="bicubic", align_corners=False)
                bbox_tensor = torch.tensor([[x1, y1, x2, y2]], device=self.device)
                composite_tensor = overlay_patch_rgba(composite_tensor, patch_resized, bbox_tensor, alpha=1.0)

            # --- Save composite image for inspection ---
            self.dataset.save_composite(composite_tensor, f"composite_step_{step+1}.png")

            # --- Prepare input for YOLO ---
            composite_for_yolo = composite_tensor.clamp(0, 1).to(self.device)
            if composite_for_yolo.shape[-2:] != (640, 640):
                composite_for_yolo = F.interpolate(composite_for_yolo, size=(640, 640), mode="bilinear", align_corners=False)
            if composite_for_yolo.dim() == 3:
                composite_for_yolo = composite_for_yolo.unsqueeze(0)  # [1, 3, 640, 640]

            # --- Prepare YOLO label batch dict ---

            h0, w0 = orig_img_tensor.shape[-2:]  # original image size
            r = min(640 / h0, 640 / w0)
            new_unpad = (int(round(w0 * r)), int(round(h0 * r)))
            dw = (640 - new_unpad[0]) / 2  # width padding
            dh = (640 - new_unpad[1]) / 2  # height padding
            class_ids, bboxes = [], []
            for bbox in orig_bboxes:
                class_id = int(bbox[0][0])
                x1, y1, x2, y2 = [float(bbox[j][0]) for j in range(1, 5)]
                # Scale
                x1 = x1 * r + dw
                x2 = x2 * r + dw
                y1 = y1 * r + dh
                y2 = y2 * r + dh
                # Normalize
                x1 /= 640
                x2 /= 640
                y1 /= 640
                y2 /= 640
                class_ids.append([class_id])
                bboxes.append([x1, y1, x2, y2])
            if len(class_ids) == 0:
                class_ids = torch.zeros((0, 1), device=self.device)
                bboxes = torch.zeros((0, 4), device=self.device)
            else:
                class_ids = torch.tensor(class_ids, device=self.device, dtype=torch.float32)
                bboxes = torch.tensor(bboxes, device=self.device, dtype=torch.float32)

            batch_dict = {
                "batch_idx": torch.zeros((class_ids.shape[0], 1), device=self.device, dtype=torch.float32),
                "cls": class_ids,
                "bboxes": bboxes,
            }
            # batch_dict['bboxes'] = torch.tensor([[0.4, 0.4, 0.6, 0.6]], device=self.device)
            # batch_dict['cls'] = torch.tensor([[0]], device=self.device)
            # batch_dict['batch_idx'] = torch.zeros((1, 1), device=self.device)
            # --- Ensure YOLOv8 loss hyperparameters are present ---
            default_hyp = {"box": 0.05, "cls": 0.5, "dfl": 1.5}
            for k, v in default_hyp.items():
                if k not in self.yolo.model.args:
                    self.yolo.model.args[k] = v

            # --- Forward through YOLO and compute loss ---
            self.yolo.model.train()
            preds = self.yolo.model(composite_for_yolo)
            loss, _ = self.yolo_loss(preds, batch_dict)

            # --- Backpropagation and logging ---
            loss_scalar = loss.sum().item()
            loss_history.append(loss_scalar)
            optimizer.zero_grad()
            (-loss.sum()).backward()

            # Loss breakdown (box, cls, dfl)
            box_loss = loss[0].item() if loss.numel() > 0 else float('nan')
            cls_loss = loss[1].item() if loss.numel() > 1 else float('nan')
            dfl_loss = loss[2].item() if loss.numel() > 2 else float('nan')

            # Latent statistics
            z_norm = z.norm().item()
            z_min = z.min().item()
            z_max = z.max().item()
            z_mean = z.mean().item()

            # Gradient statistics
            grad_norm = z.grad.norm().item() if z.grad is not None else 0.0
            grad_min = z.grad.min().item() if z.grad is not None else float('nan')
            grad_max = z.grad.max().item() if z.grad is not None else float('nan')
            grad_mean = z.grad.mean().item() if z.grad is not None else float('nan')

            self.logger.info(
                f"Step {step+1:03d} | "
                f"Loss: {loss_scalar:.4f} (box: {box_loss:.4f}, cls: {cls_loss:.4f}, dfl: {dfl_loss:.4f}) | "
                f"z norm: {z_norm:.4f} | "
                f"z grad norm: {grad_norm:.4f} | z.grad min: {grad_min:.4f} | z.grad max: {grad_max:.4f} | z.grad mean: {grad_mean:.4f}"
            )

            # --- Plot and save gradient histogram ---
            plt.figure()
            plt.hist(z.grad.detach().cpu().numpy().flatten(), bins=50)
            plt.title(f"z.grad histogram step {step+1}")
            plt.xlabel("Gradient value")
            plt.ylabel("Frequency")
            plt.tight_layout()
            plt.savefig(f"z_grad_hist_step_{step+1}.png")
            plt.close()

            optimizer.step()

        # --- Save optimized latent vector ---
        torch.save(z.detach().cpu(), "optimized_latent.pt")
        self.logger.info("Optimized latent vector saved to optimized_latent.pt")

        # --- Plot loss curve ---
        plt.figure()
        plt.plot(loss_history, label="Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title("Latent Optimization Loss Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig("latent_optimization_loss_curve.png")
        plt.close()
        self.logger.info("Loss curve saved to latent_optimization_loss_curve.png")