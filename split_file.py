from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split


def split_dataset(source_dir: Path, output_dir: Path):
    """
    Split the dataset into training, validation sets.
    """
    image_exts = {".jpg", ".jpeg"}

    all_files = []
    all_labels = []

    for class_dir in source_dir.iterdir():
        if class_dir.is_dir() and class_dir.name != "splited":
            for file in class_dir.iterdir():
                if file.is_file() and file.suffix.lower() in image_exts:
                    all_files.append(file)
                    all_labels.append(class_dir.name)

    # Split: 80% train, 20% val
    train_files, val_files, train_labels, val_labels = train_test_split(
        all_files,
        all_labels,
        test_size=0.2,
        random_state=42,
        stratify=all_labels
    )

    copy_files(train_files, train_labels, "train", output_dir)
    copy_files(val_files, val_labels, "val", output_dir)

    print(f"Train: {len(train_files)}")
    print(f"Val:   {len(val_files)}")
    print(f"Done. Files copied to: {output_dir}")


def copy_files(files: list, labels: list, split_name: str, output_dir: Path):
    """
    Copy files into output_dir/split_name/class_name/
    """
    for file_path, label in zip(files, labels):
        class_output_dir = output_dir / split_name / label
        class_output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, class_output_dir / file_path.name)
