import sys
import shutil
import cv2
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import yaml
import tensorflow as tf
from utils import is_path_dir
import pandas as pd
from split_file import split_dataset
from Transformation import transformation


def load_config(config_path: Path = Path("configs/train.yaml")) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def configure_device():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU detected ({len(gpus)} device(s)) — training on GPU.")
    else:
        print("No GPU detected — training on CPU.")


def train_tf(source_dir: Path, config: dict):
    """
    Train a TensorFlow model on the dataset.
    """
    output_dir = source_dir / "splited"
    split_dataset(source_dir, output_dir)

    for item in source_dir.iterdir():
        if item.is_dir() and item.name != "splited":
            shutil.rmtree(item)

    train_set = output_dir / "train"
    val_set = output_dir / "val"

    image_size = tuple(config["image_size"])
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    es_cfg = config["early_stopping"]
    logs_dir = config.get("logs_dir", "logs/training")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_set,
        image_size=image_size,
        batch_size=batch_size,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_set,
        image_size=image_size,
        batch_size=batch_size,
    )

    class_names = train_ds.class_names

    prefetch = tf.data.AUTOTUNE if config.get("autotune", True) else 2
    train_ds = train_ds.prefetch(prefetch)
    val_ds = val_ds.prefetch(prefetch)

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(*image_size, 3)),
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(len(class_names), activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor=es_cfg["monitor"],
        patience=es_cfg["patience"],
        restore_best_weights=es_cfg["restore_best_weights"],
    )

    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=logs_dir,
        histogram_freq=1,
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[early_stop, tensorboard],
    )

    _, accuracy = model.evaluate(val_ds)
    print("Validation accuracy:", accuracy)

    df = pd.DataFrame(class_names, columns=["class_name"])
    class_names_path = output_dir / "class_names.csv"
    df.to_csv(class_names_path, index=False)

    model_path = output_dir / "leaf_model.keras"
    model.save(model_path)
    print("Model saved to:", model_path)

    shutil.rmtree(output_dir / "train", ignore_errors=True)
    shutil.rmtree(output_dir / "val", ignore_errors=True)


def _transform_one(file_path: Path, dest_class_dir: Path):
    transformed = transformation(file_path)
    for trn_name, trn_img in transformed.items():
        save_path = dest_class_dir / f"{file_path.stem}_{trn_name}.JPG"
        cv2.imwrite(str(save_path), trn_img)


def transformation_dir(src_dir_path: Path, dest_dir_path: Path):
    """
    Apply transformations to all images in the source directory
    and save them to the destination directory.
    Uses rglob so it works regardless of nesting depth — the immediate
    parent folder of each image is used as the class name.
    """
    tasks = []
    for file_path in src_dir_path.rglob("*.JPG"):
        class_name = file_path.parent.name
        dest_class_dir = dest_dir_path / class_name
        dest_class_dir.mkdir(parents=True, exist_ok=True)
        tasks.append((file_path, dest_class_dir))

    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(_transform_one, fp, dd): fp for fp, dd in tasks
        }
        for future in as_completed(futures):
            future.result()


def main():
    """main()"""

    try:

        if len(sys.argv) != 3:
            print("Error: the arguments are bad")
            return

        to_train_dirpath = Path(sys.argv[1])
        is_path_dir(to_train_dirpath)

        transformed_dirpath = Path(sys.argv[2])
        config = load_config()

        configure_device()
        transformation_dir(to_train_dirpath, transformed_dirpath)
        train_tf(transformed_dirpath, config)

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
