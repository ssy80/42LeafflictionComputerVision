# Leaffliction

*42 Advanced — Computer Vision project by ssian and axlee.*

Classifies plant leaf diseases from images using a CNN trained on Apple and Grape leaves across 8 disease classes.

## Dataset Classes

| Apple | Grape |
|---|---|
| Apple\_Black\_rot | Grape\_Black\_rot |
| Apple\_healthy | Grape\_Esca |
| Apple\_rust | Grape\_healthy |
| Apple\_scab | Grape\_spot |

---

## Project Structure

```
Leaffliction/
├── Distribution.py       ← pie + bar charts of class balance
├── Augmentation.py       ← apply 6 augmentations to a single image
├── Transformation.py     ← apply 6 PlantCV transformations + histogram
├── train.py              ← train CNN, save model + learnings zip
├── predict.py            ← classify a leaf image
├── utils.py              ← shared validation utilities
├── split_file.py         ← 80/20 stratified train/val split
├── scripts/
│   ├── setup.sh          ← create venv, install dependencies
│   ├── remove.sh         ← tear down venv and all generated data
│   ├── augmentation.sh   ← batch augment + balance all classes
│   ├── transformation.sh ← batch transform augmented_directory/
│   ├── train.sh          ← train Apple and Grape models in sequence
│   └── lint.sh           ← run flake8 on all source files
├── requirements.txt
├── en.subject.pdf
└── test/                 ← sample images for prediction
    ├── Apple/
    └── Grape/
```

---

## Setup

Place your dataset at the project root:

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

Then install the environment:

```bash
source scripts/setup.sh
```

> On machines with an NVIDIA GPU, setup automatically installs CUDA-enabled TensorFlow and patches `LD_LIBRARY_PATH`. On CPU-only machines it installs plain TensorFlow.

If GPU is not detected after setup, re-source the activate script:

```bash
deactivate && source my_env/bin/activate
```

---

## Full Pipeline

```bash
source scripts/setup.sh                              # 1. set up environment
./Distribution.py ./Apple                            # 2. analyse raw dataset
./scripts/augmentation.sh                            # 3. augment + balance → augmented_directory/
./Distribution.py ./augmented_directory              # 4. verify class balance
./scripts/train.sh                                   # 5. train Apple + Grape models
./predict.py "./Apple/Apple_healthy/image (1).JPG"   # 6. predict (auto-detects model)
```

---

## Command Reference

### Part 1 — Distribution

Displays pie and bar charts for each class in a dataset directory.

```bash
./Distribution.py <directory>
```

```bash
./Distribution.py ./Apple
./Distribution.py ./Grape
./Distribution.py ./augmented_directory
```

---

### Part 2 — Augmentation

Applies 6 augmentations to a single image (Flip, Rotate, Skew, Contrast, Crop, Distortion). Saves results alongside the original as `<stem>_Flip.JPG`, etc.

```bash
./Augmentation.py <image_path>
```

```bash
./Augmentation.py "./Apple/Apple_healthy/image (1).JPG"
```

To augment the full dataset, balance all 8 classes to the same count, and create `augmented_directory/`:

```bash
./scripts/augmentation.sh
```

---

### Part 3 — Transformation

Applies 6 PlantCV transformations to a single image and displays a colour histogram.

```bash
./Transformation.py <image_path>
```

```bash
./Transformation.py "./Apple/Apple_healthy/image (1).JPG"
```

To save transformations for a whole directory:

```bash
./Transformation.py -src <source_dir> -dst <dest_dir>
```

To transform all the images in the directory:

```bash
./scripts/transformation.sh
```

To see the progress of transformation:

```bash
find ./augmented_directory/Apple -name "*.JPG" | wc -l
```

To see progress per class:

```bash
for d in ./augmented_directory/Apple/*/; do echo "$(find "$d" -name "*.JPG" | wc -l) $(basename $d)"; done

for d in ./augmented_directory/Grape/*/; do echo "$(find "$d" -name "*.JPG" | wc -l) $(basename $d)"; done
```
---

### Part 4 — Training

Trains a CNN on a labelled image directory. Splits 80/20 train/val, trains with early stopping, and saves the model plus a learnings zip.

```bash
./train.py <dataset_dir>
```

```bash
./train.py ./augmented_directory/Apple   # → augmented_directory/Apple_learnings.zip
./train.py ./augmented_directory/Grape   # → augmented_directory/Grape_learnings.zip
```

To train both in one go:

```bash
./scripts/train.sh
```

**Outputs per model:**
- `<dataset_dir>/splited/leaf_model.keras`
- `<dataset_dir>/splited/class_names.csv`
- `<dataset_dir>/../<name>_learnings.zip`

GPU is detected automatically — no flags needed.

---

### Part 5 — Prediction

Classifies a leaf image and displays the original alongside its masked transformation with the predicted class label.

```bash
./predict.py <image_path> [model_dir]
```

The model is auto-detected from `Apple` or `Grape` in the image path. Pass `model_dir` explicitly if needed.

```bash
./predict.py "./Apple/Apple_healthy/image (1).JPG"
./predict.py "./test/Apple/image (1).JPG" augmented_directory/Apple/splited
```

---

## Turn-in

After training, generate `signature.txt` from the learnings zip:

```bash
sha1sum augmented_directory/Apple_learnings.zip > signature.txt
```

Commit `signature.txt` to the repository. During evaluation, the hash will be verified against the zip — they must match exactly.

> Do **not** commit the dataset or zip files. Do **not** retrain after generating `signature.txt`.

---

## Resources

- [Subject PDF](en.subject.pdf)
- [TensorFlow — Image classification](https://www.tensorflow.org/tutorials/images/classification)
- [PlantCV documentation](https://plantcv.readthedocs.io/)
