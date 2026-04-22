import sys
import cv2
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from training.model import load_config, configure_device, train_tf
from transformation.transformation import transformation
from utils.utils import is_path_dir


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
