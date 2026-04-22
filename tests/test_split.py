from split.split_file import split_dataset


def test_split_creates_train_and_val(sample_class_dir, tmp_path):
    out = tmp_path / "split"
    split_dataset(sample_class_dir, out)
    assert (out / "train").exists()
    assert (out / "val").exists()


def test_split_preserves_all_classes(sample_class_dir, tmp_path):
    out = tmp_path / "split"
    split_dataset(sample_class_dir, out)
    train_classes = {p.name for p in (out / "train").iterdir() if p.is_dir()}
    val_classes = {p.name for p in (out / "val").iterdir() if p.is_dir()}
    assert train_classes == {"class_a", "class_b"}
    assert val_classes == {"class_a", "class_b"}


def test_split_80_20_ratio(sample_class_dir, tmp_path):
    out = tmp_path / "split"
    split_dataset(sample_class_dir, out)
    total_train = sum(len(list(d.glob("*.JPG"))) for d in (out / "train").iterdir() if d.is_dir())
    total_val = sum(len(list(d.glob("*.JPG"))) for d in (out / "val").iterdir() if d.is_dir())
    # fixture: 4 images × 2 classes = 8 total → 6 train, 2 val
    assert total_train + total_val == 8
    assert total_train == 6
    assert total_val == 2


def test_split_no_duplicates(sample_class_dir, tmp_path):
    out = tmp_path / "split"
    split_dataset(sample_class_dir, out)
    train_files = {p.name for d in (out / "train").iterdir() if d.is_dir() for p in d.glob("*.JPG")}
    val_files = {p.name for d in (out / "val").iterdir() if d.is_dir() for p in d.glob("*.JPG")}
    assert train_files.isdisjoint(val_files)


def test_split_model_predict(sample_class_dir, tmp_path):
    """Train a tiny model and verify predict runs without error."""
    import tensorflow as tf
    import numpy as np

    out = tmp_path / "split"
    split_dataset(sample_class_dir, out)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        out / "train", image_size=(64, 64), batch_size=4
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        out / "val", image_size=(64, 64), batch_size=4
    )

    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255, input_shape=(64, 64, 3)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(len(train_ds.class_names), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_ds, epochs=1, verbose=0)

    dummy = np.zeros((1, 64, 64, 3), dtype=np.float32)
    pred = model.predict(dummy, verbose=0)
    assert pred.shape == (1, len(train_ds.class_names))
