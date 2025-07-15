# Graffiti Attack

Adversarial Patch Optimization Using StyleGAN3, YOLOv8, and Nevergrad

This project investigates the use of generative models and black-box optimization to produce naturalistic adversarial patches in the form of graffiti, aimed at degrading the performance of object detection systems in autonomous driving scenarios.
Generator: StyleGAN3 (Fine-tuned on Graffiti Data)

The generator is based on StyleGAN3, fine-tuned on a curated dataset of approximately 1,000 graffiti images. StyleGAN3 is selected for:

    Its high style diversity, which enables generation of visually diverse graffiti patterns that better mimic real-world variability.

    Its use of Adaptive Data Augmentation (ADA), which improves training robustness in low-data regimes.

Patch Extraction and Application

Generated graffiti samples are post-processed using OpenCV to extract the relevant patch region, which is then overlaid onto traffic sign images. Currently, patches are applied at the center of the sign area for controlled testing.

    Future Work: Integrate a placement optimization module to allow spatial variation and improve attack strength.

Adversarial Optimization Strategy

The target model is YOLOv8m, treated as a black-box detector. Since gradient information is unavailable, Nevergrad, an evolutionary optimization framework, is employed to search the StyleGAN3 latent space for graffiti samples that maximize adversarial effect.

Two attack objectives are considered:

    Detection Attack: Minimize the detector's confidence score or suppress detection entirely.

    Classification Attack: Induce misclassification (e.g., altering predictions from "Stop Sign" to "Speed Limit").

Planned Enhancements

    Surrogate Model Integration: A white-box surrogate will be incorporated during GAN training to embed adversarial priors directly into the generator, improving transferability and attack reliability.

    Cross-model Evaluation: Additional object detectors will be included to assess generalization and robustness of the optimized patches across architectures.
---

## Project Structure

```
graffiti_attack/
├──src
    ├── models.py          # Loads StyleGAN3 and YOLOv8 models
    ├── patch.py           # Patch creation, masking, and application logic
    ├── optimization.py    # Evolution strategy and loss functions
    ├── utils.py           # Helper utilities
    ├── main.py            # Main experiment loop
    ├── test_models.py     # Sanity check for model loading
├── requirements.txt   # Dependencies (excludes StyleGAN3)
└── README.md          # This file
```

## Notes

This work is part of my Master's thesis project conducted at Waseda University under the supervision of Prof. Tatsuya Mori.

---

## License

- This repo: [MIT License](LICENSE)
- StyleGAN3: [NVlabs License](https://github.com/NVlabs/stylegan3/blob/main/LICENSE.txt)
