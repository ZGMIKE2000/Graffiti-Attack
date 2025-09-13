# Graffiti Attack
## Adversarial Patch Optimization Using GAN

This project explores the creation of naturalistic adversarial patches in the form of graffiti to disrupt object detection models used in autonomous driving scenarios. The goal is to generate visually realistic graffiti that can reduce detection confidence or suppress detection entirely on traffic signs, particularly Stop signs, which are critical for safety.

## Project Overview

The attack follows a two-step approach:

- White-box fine-tuning of a generator: Learning both natural graffiti styles and adversarial features using a differentiable object detector.

- Black-box latent space search: Using evolutionary optimization to identify the most effective patch parameters and latent codes for physical deployment.

The combination of these steps ensures that patches are both natural-looking and adversarially effective in real-world conditions.


## Generator: StyleGAN3

We use StyleGAN3, fine-tuned on a curated dataset of ~1,000 graffiti images. StyleGAN3 was chosen for:

High style diversity: Can produce a wide variety of realistic graffiti patterns.

Adaptive Data Augmentation (ADA): Improves robustness under limited data conditions.

Two-step Fine-tuning

- Stage 1 – Natural Graffiti Learning

The generator is trained on graffiti images to learn the underlying style distribution.

Focus: Naturalness, ensuring generated patches resemble real-world graffiti.

- Stage 2 – White-box Adversarial Fine-tuning

Using a YOLOv8 detector as a white-box, we calculate gradients w.r.t objectness and class probabilities.

Gradients are backpropagated through the generator to imbue patches with adversarial features.

All transformations and augmentations in this stage are fully differentiable, allowing end-to-end gradient optimization.

## Patch Extraction and Overlay

Generated graffiti samples are post-processed using OpenCV to extract the patch region.

Patches are overlaid onto traffic sign images with augmentations from the Kornia library.

Losses are calculated by comparing the model’s predictions on patched vs. original images.

Note: In black-box experiments, only non-differentiable CV operations are used—no gradients are accessed.

## Black-box Optimization: Latent Search with Nevergrad

To optimize patches for physical deployment:

We treat the YOLO model as a black-box, with no gradient access.

CMA-ES (Covariance Matrix Adaptation) from Nevergrad is used to search the StyleGAN3 latent space.

Optimization jointly searches for:

Best latent code for patch generation.

Optimal patch position, scale, and transparency on the traffic sign.

This iterative search allows us to identify patches that are robust and adversarially effective in real-world conditions.


## Experiment Summary

Patches covering ~50–90% of a sign can disrupt YOLO detection.

Smaller patches (<50%) generally require careful optimization of position, alpha, and scale.

Current experiments focus on Stop signs, with plans to extend to other classes for benchmarking.

---

## Project Structure

```
graffiti_attack/
├── src/
│   ├── whitebox/
│   │   ├── eot.py             # EOT augmentation pipeline from Kornia
│   │   ├── main_whitebox.py   # Main script for white-box fine-tuning
│   │   ├── models.py          # Loads StyleGAN3 and YOLOv8 models (white-box)
│   │   ├── patch.py           # Patch creation, masking, and application logic
│   │   ├── placement.py       # Helper for patching mechanism
│   │   ├── processing.py      # Helper for white-box optimization (gradients, losses)
│   │
│   ├── blackbox/
│   │   ├── main_blackbox.py   # Main script for black-box latent search
│   │   ├── optimization.py    # Evolution strategy (CMA-ES) and loss functions
│   │   ├── patch.py           # Patch creation, masking, and application logic (non-differentiable)
│   │   ├── placement.py       # Helper for patching mechanism
│   │
│   ├── utils.py               # Shared helper utilities
│   ├── test_models.py         # Sanity check for model loading
├── requirements.txt           # Dependencies (excludes StyleGAN3)
└── README.md                  # This file
```

## Notes

This work is part of my Master's thesis project conducted at Waseda University under the supervision of Prof. Tatsuya Mori, in collaboration with Politecnico di Milano under the supervision of Prof. Stefano Zanero.

---

## License

- This repo: [MIT License](LICENSE)
- StyleGAN3: [NVlabs License](https://github.com/NVlabs/stylegan3/blob/main/LICENSE.txt)
