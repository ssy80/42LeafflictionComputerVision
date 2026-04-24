#!/usr/bin/env my_env/bin/python3
import sys
from pathlib import Path
import tensorflow as tf
from utils import is_path_dir
import pandas as pd
from split_file import split_dataset
from Transformation import transform_dir


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

    image_size = (256, 256)
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

    model = tf.keras.Sequential([

        tf.keras.layers.Rescaling(1./255, input_shape=(*image_size, 3)),

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

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
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


def transformation_dir(src_dir_path: Path, dest_dir_path: Path):
    """
    Apply transformations to all images in the source directory
    and save them to the destination directory.
    """
    for class_dir in src_dir_path.iterdir():
        if class_dir.is_dir():
            dest_class_dir = dest_dir_path / class_dir.name
            dest_class_dir.mkdir(parents=True, exist_ok=True)
            transform_dir(class_dir, dest_class_dir)


def main():
    try:

        if len(sys.argv) != 3:
            print("Error: the arguments are bad")
            return

        source_dir = Path(sys.argv[1])
        is_path_dir(source_dir)

        transformed_dirpath = Path(sys.argv[2])
        transformation_dir(source_dir, transformed_dirpath)

        configure_device()
        train_tf(transformed_dirpath)

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
