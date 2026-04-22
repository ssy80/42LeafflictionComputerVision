# Leaffliction

*This project has been created as part of the 42 Advanced curriculum by ssian and axlee.*

## Description

Leaffliction is a machine learning project that classifies plant leaf diseases from images. Given a photo of an apple or grape leaf, the model identifies which disease (if any) is present across 8 classes.

The pipeline covers the full ML workflow:

* **Distribution** — visualise the class balance of a dataset as pie and bar charts.
* **Augmentation** — expand the dataset by applying 6 transformations per image (Flip, Rotate, Skew, Contrast, Crop, Distortion), saving augmented variants alongside the original.
* **Transformation** — apply computer vision preprocessing (Original, Gaussian blur, Mask, ROI, Analyze, Pseudolandmarks) and display a colour histogram.
* **Training** — split 80/20 and train a CNN (TensorFlow/Keras) on the augmented dataset with GPU support. Model saved to `<output_dir>/splited/`.
* **Prediction** — classify a single leaf image and display the original alongside its masked transformation and predicted disease label.

### Dataset Classes

| Apple | Grape |
|---|---|
| Apple\_Black\_rot | Grape\_Black\_rot |
| Apple\_healthy | Grape\_Esca |
| Apple\_rust | Grape\_healthy |
| Apple\_scab | Grape\_spot |

## Project Structure

```
Leaffliction/
├── Distribution.py     ← analyse dataset class balance (pie + bar charts)
├── Augmentation.py     ← apply 6 augmentations to a single image
├── Transformation.py   ← apply 6 CV transformations + colour histogram
├── train.py            ← transform images, train CNN, save model
├── predict.py          ← load model and predict a leaf image
├── utils.py            ← shared path/file validation utilities
├── split_file.py       ← 80/20 train/val split helper
├── scripts/
│   ├── setup.sh           ← environment setup (venv + dependencies)
│   ├── remove.sh          ← tear down environment and generated data
│   ├── augmentation.sh    ← batch augmentation across all classes
│   └── transformation.sh  ← batch transformation of augmented_directory/
├── requirements.txt
├── en.subject.pdf
└── test/               ← test images for prediction
    ├── Apple/
    └── Grape/
```

## Setup

Place your dataset images in a directory following this structure:

```
Apple/
├── Apple_Black_rot/
├── Apple_healthy/
├── Apple_rust/
└── Apple_scab/
Grape/
├── Grape_Black_rot/
├── Grape_Esca/
├── Grape_healthy/
└── Grape_spot/
```

Then run:

```bash
source scripts/setup.sh
```

> On machines with an NVIDIA GPU, `scripts/setup.sh` automatically installs CUDA-enabled TensorFlow and configures the library paths. On CPU-only machines (e.g. school PCs) it installs plain TensorFlow.

---

## Order of Execution

```bash
source scripts/setup.sh                                       # 1. set up environment
./Distribution.py ./Apple                                     # 2. analyse raw dataset
./scripts/augmentation.sh                                     # 3. augment and balance → augmented_directory/
./Distribution.py ./augmented_directory                       # 4. verify balance after augmentation
./train.py ./augmented_directory/Apple                        # 5. train Apple model → Apple_learnings.zip
./train.py ./augmented_directory/Grape                        # 6. train Grape model → Grape_learnings.zip
./predict.py "./Apple/Apple_healthy/image (1).JPG"            # 7. predict (auto-detects model)
```

> **Note — Transformation is a standalone deliverable (Part 3), not a preprocessing step for training.**
> Run `./Transformation.py <image>` to visualise PlantCV analysis on a single leaf.
> The CNN is trained directly on the original and augmented images.

---

## Command Reference

### Part 1 — Distribution

Analyse a dataset directory and display pie + bar charts for each class.

```bash
./Distribution.py <directory>
```

Examples:
```bash
./Distribution.py ./Apple
./Distribution.py ./Grape
```

---

### Part 2 — Augmentation

Apply 6 augmentations to a single image. Augmented files are saved alongside the original, named `<stem>_Flip.JPG`, `<stem>_Rotate.JPG`, etc.

```bash
./Augmentation.py <image_path>
```

Example:
```bash
./Augmentation.py "./Apple/Apple_healthy/image (1).JPG"
```

To batch-augment and balance the full dataset across all classes:

```bash
./scripts/augmentation.sh
```

This augments every original image with 6 variants (Flip, Rotate, Skew, Contrast, Crop, Distortion), trims all classes to the same count, and creates `augmented_directory/` — a flat directory containing all 8 class folders, ready for distribution analysis:

```bash
./Distribution.py ./augmented_directory/Apple
./Distribution.py ./augmented_directory/Grape
```

---

### Part 3 — Transformation

Apply 6 computer vision transformations to an image and display the results alongside a colour histogram.

```bash
./Transformation.py <image_path>
```

Example:
```bash
./Transformation.py "./Apple/Apple_healthy/image (1).JPG"
```

To batch-transform a single directory and save results:

```bash
./Transformation.py -src <source_dir> -dst <dest_dir>
```

To batch-transform all classes across Apple and Grape into `augmented_directory/Apple/` and `augmented_directory/Grape/`:

```bash
./scripts/transformation.sh
```

---

### Part 4 — Training

Train a CNN on an augmented labelled image directory, then package the model and class names into a zip.

```bash
./train.py <transformed_dir>
```

If GPU is not found
```bash
grep -n "nvidia" my_env/bin/activate

deactivate && source my_env/bin/activate

```

To batch train both Apple and Grape
```bash
./scripts/train.sh
```

Outputs:
- `<transformed_dir>/splited/leaf_model.keras` — trained model
- `<transformed_dir>/splited/class_names.csv` — class label map
- `<transformed_dir>/../<name>_learnings.zip` — zip containing model files and all transformed images

Examples:
```bash
./train.py ./augmented_directory/Apple   # → augmented_directory/Apple_learnings.zip
./train.py ./augmented_directory/Grape   # → augmented_directory/Grape_learnings.zip
```

GPU is detected automatically — no extra flags needed.

---

### Part 5 — Prediction

Classify a leaf image using a trained model. Displays the original and masked transformation with the predicted class label.

```bash
./predict.py <image_path> [model_dir]
```

If `model_dir` is omitted, the model is auto-detected from the image path (`Apple` or `Grape`).

Examples:
```bash
./predict.py "./Apple/Apple_healthy/image (1).JPG"                      # auto-detect
./predict.py "./test/Apple/image (1).JPG" augmented_directory/Apple/splited   # explicit
./predict.py "./test/Grape/image (1).JPG" augmented_directory/Grape/splited
```

---

## Resources

* [42 Curriculum — Leaffliction subject](en.subject.pdf)
* [TensorFlow — Image classification guide](https://www.tensorflow.org/tutorials/images/classification)
* [PlantCV documentation](https://plantcv.readthedocs.io/)
* [Wikipedia: Convolutional Neural Network](https://en.wikipedia.org/wiki/Convolutional_neural_network)
