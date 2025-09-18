import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np

class StopSignBBoxDataset(Dataset):
    """
    PyTorch Dataset for stop sign images and bounding boxes in YOLO format.
    Loads images and their corresponding bounding box labels.
    """

    def __init__(self, image_dir, label_dir, image_size=640):
        """
        Args:
            image_dir (str): Directory containing images.
            label_dir (str): Directory containing YOLO-format label .txt files.
            image_size (int): Size to which images are resized.
        """
        self.image_paths = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".png"))
        ]
        self.label_dir = label_dir
        self.image_size = image_size
        self.to_tensor = T.ToTensor()
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def _load_label(self, img_path, orig_w, orig_h):
        """
        Loads bounding boxes for an image from its label file.
        Returns a list of (class, x1, y1, x2, y2) tuples in pixel coordinates.
        """
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(self.label_dir, base + ".txt")
        if not os.path.exists(label_path):
            return None

        bboxes = []
        with open(label_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls, x, y, w, h = map(float, parts)
                # Convert YOLO (cx, cy, w, h) normalized to pixel (x1, y1, x2, y2)
                x, y, w, h = x * orig_w, y * orig_h, w * orig_w, h * orig_h
                x1 = int(x - w/2)
                y1 = int(y - h/2)
                x2 = int(x + w/2)
                y2 = int(y + h/2)
                bboxes.append((int(cls), x1, y1, x2, y2))
        return bboxes

    def __getitem__(self, idx):
        """
        Returns a dict with:
            - 'orig_img': image as a tensor (unnormalized)
            - 'orig_bboxes': list of (class, x1, y1, x2, y2)
            - 'orig_size': (width, height) of the original image
        """
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size
        bboxes = self._load_label(img_path, orig_w, orig_h)
        img_tensor = self.to_tensor(img)  # Unnormalized tensor
        return {
            "orig_img": img_tensor,
            "orig_bboxes": bboxes,
            "orig_size": (orig_w, orig_h)
        }

    def resize_for_yolo(self, pil_img):
        """
        Resize a PIL image to the YOLO input size and apply normalization.
        Returns a normalized tensor.
        """
        img_resized = pil_img.resize((self.image_size, self.image_size), Image.BILINEAR)
        img_tensor = self.transform(img_resized)
        return img_tensor

    def tensor_to_pil(self, tensor_img):
        """
        Converts a tensor image (C,H,W) to a PIL Image.
        """
        np_img = tensor_img.detach().cpu().squeeze(0).permute(1,2,0).clamp(0,1).numpy()
        return Image.fromarray((np_img * 255).astype(np.uint8))

    def save_composite(self, tensor_img, save_path=None):
        """
        Saves a tensor image as a composite PNG (for inspection).
        """
        pil_img = self.tensor_to_pil(tensor_img)
        if save_path is not None:
            pil_img.save(save_path)
        return pil_img