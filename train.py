#!/usr/bin/env my_env/bin/python3
import sys
import shutil
import zipfile
from pathlib import Path
import tensorflow as tf
from utils import is_path_dir
import pandas as pd
from split_file import split_dataset


def configure_device():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU detected ({len(gpus)} device(s)) — training on GPU.")
    else:
        print("No GPU detected — training on CPU.")


def train_tf(source_dir: Path):
    output_dir = source_dir / "splited"
    split_dataset(source_dir, output_dir)

    train_set = output_dir / "train"
    val_set = output_dir / "val"

    image_size = (128, 128)
    batch_size = 32

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_set,
        image_size=image_size,
        batch_size=batch_size,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_set,
        image_size=image_size,
        batch_size=batch_size
    )

    val_count = sum(1 for _ in val_ds.unbatch())
    print(f"Validation set size: {val_count} images")
    if val_count < 100:
        print(
            f"Warning: validation set has fewer than 100 images "
            f"({val_count}). Augment your dataset further."
        )

    model = tf.keras.Sequential([

        tf.keras.layers.Rescaling(1./255, input_shape=(*image_size, 3)),

        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),

        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Dropout(0.3),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(len(train_ds.class_names), activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=5, restore_best_weights=True
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        callbacks=[early_stop]
    )

    _, accuracy = model.evaluate(val_ds)
    print("Validation accuracy:", accuracy)

    class_names = train_ds.class_names
    df = pd.DataFrame(class_names, columns=["class_name"])
    class_names_path = output_dir / "class_names.csv"
    df.to_csv(class_names_path, index=False)

    model_path = output_dir / "leaf_model.keras"
    model.save(model_path)
    print("Model saved to:", model_path)

    zip_path = source_dir.parent / f"{source_dir.name}_learnings.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(model_path, arcname=f"splited/{model_path.name}")
        zf.write(class_names_path, arcname=f"splited/{class_names_path.name}")
        for class_dir in source_dir.iterdir():
            if class_dir.is_dir() and class_dir.name != "splited":
                for img_file in class_dir.glob("*.JPG"):
                    zf.write(img_file,
                             arcname=f"{class_dir.name}/{img_file.name}")
    print("Learnings saved to:", zip_path)

    shutil.rmtree(output_dir / "train", ignore_errors=True)
    shutil.rmtree(output_dir / "val", ignore_errors=True)


def main():
    try:

        if len(sys.argv) not in (2, 3):
            print("Error: the arguments are bad")
            return

        source_dir = Path(sys.argv[1])
        is_path_dir(source_dir)

        configure_device()
        train_tf(source_dir)

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
