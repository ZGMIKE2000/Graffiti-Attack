# Graffiti Attack 

Adversarial graffiti patch pipeline that combines (1) white‑box latent optimization using StyleGAN3 and a differentiable detector, and (2) black‑box patch‑parameter search (Nevergrad) for realistic placement and transferability across detectors.

## Overview
- White‑box: optimize a latent vector in StyleGAN3 so generated patches reduce detector confidence when overlaid (requires gradient access to the detector).
- Black‑box: given a patch image (GAN output or static PNG), optimize non‑gradient parameters — scale, position, alpha, etc. — using an evolutionary optimizer (Nevergrad/CMA‑ES) and evaluate against one or an ensemble of detectors (inference only).
- Purpose: produce realistic patches that are adversarial and robust across different detectors and scenes.

## Methodology
First, use white‑box latent optimization to produce visually realistic adversarial patches (end‑to‑end differentiable). Export one or more patch PNGs. Second, run black‑box parameter optimization that treats detectors as inference oracles and searches for placement parameters that maximize transferability (minimize ensemble detection confidence). This two‑stage separation keeps generation (content) and placement (physical realism / transferability) decoupled.

## Quick usage
1. Generate candidate patches via white‑box trainer and save PNG(s).
2. Run black‑box optimizer with a patch, target image(s), and one or more YOLO checkpoints:
```bash
python src/blackbox/main.py \
  --patch /path/to/patch.png \
  --img /path/to/img.jpg \
  --yolo /path/to/yolov8.pt \
  --target-class 0 \
  --generations 50 --population 8
```
Outputs: patched images, per‑evaluation log, and checkpointed best parameters.

## Practical notes
- Keep input image sizes compatible with the YOLO model (typically 640×640 or the model's training size).
- For transferability, optimize/validate on multiple images and use an ensemble of checkpoints.
- Consider adding EOT (random jitter, lighting) during black‑box evaluation for better real‑world robustness.

## What’s included
- White‑box code: generator wrapper, trainer for latent optimization (differentiable).
- Black‑box code: patch overlay utilities, detector wrapper (ensemble), Nevergrad optimizer with logging and image saving.
- Utilities: I/O, bbox parsing, checkpoint/log management.

## Dependencies
Python, PyTorch, Ultralytics YOLO, Nevergrad, OpenCV, Pillow. StyleGAN3 components required for white‑box stage only.

## License & credits
- This repo: MIT
- StyleGAN3: NVLabs license (see original repo)

For an end‑to‑end quickstart or to add EOT/extra params, say which part to automate next.
