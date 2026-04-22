import cv2
import sys
import shutil
import numpy as np
from pathlib import Path
from augmentation.transforms import flip, rotate, skew, crop, distortion
from utils.utils import is_image_file


def contrast(img: np.ndarray):
    """
    alpha = contrast factor
    beta = brightness
    """
    alpha = 1.5
    beta = 20
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def augmentation(filepath: Path, dest_dir: Path = None) -> None:
    """
    Perform six augmentations on filepath and save to dest_dir.
    If dest_dir is None, saves alongside the original.
    Also copies the original into dest_dir.
    """
    out_dir = dest_dir if dest_dir else filepath.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = filepath.stem

    shutil.copy2(filepath, out_dir / filepath.name)

    img = cv2.imread(str(filepath))

    augments = {
        "Flip": flip(img),
        "Rotate": rotate(img),
        "Skew": skew(img),
        "Contrast": contrast(img),
        "Crop": crop(img),
        "Distortion": distortion(img)
    }

    for aug_name, aug_img in augments.items():
        save_path = out_dir / f"{filename}_{aug_name}.JPG"
        cv2.imwrite(str(save_path), aug_img)


def main():
    """main()"""

    try:

        if len(sys.argv) not in (2, 3):
            print("Error: the arguments are bad")
            return

        target = Path(sys.argv[1])
        dest_dir = Path(sys.argv[2]) if len(sys.argv) == 3 else None

        if target.is_dir():
            aug_suffixes = {"Flip", "Rotate", "Skew", "Contrast", "Crop", "Distortion"}
            originals = [
                f for f in sorted(target.glob("*.JPG"))
                if not any(f.stem.endswith(f"_{s}") for s in aug_suffixes)
            ]
            print(f"Augmenting {len(originals)} images in {target.name}...")
            for filepath in originals:
                augmentation(filepath, dest_dir)
        else:
            is_image_file(target)
            augmentation(target, dest_dir)

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
