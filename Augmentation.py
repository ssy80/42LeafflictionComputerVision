#!/usr/bin/env my_env/bin/python3
import cv2
import sys
import numpy as np
from pathlib import Path
from utils import is_image_file


def flip(img: np.ndarray):
    """
    Flip image to the right
    """
    return cv2.flip(img, 1)


def rotate(img: np.ndarray):
    """
    Rotate to right 25 degree angle.
    """
    angle = -25
    h, w = img.shape[:2]
    center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(img, M, (new_w, new_h))
    return rotated


def skew(img: np.ndarray):
    """
    Skew image
    """
    h, w = img.shape[:2]
    pts1 = np.float32([[0, 0], [w, 0], [0, h]])
    pts2 = np.float32([[0, 0], [w, 0], [int(0.2*w), h]])
    M = cv2.getAffineTransform(pts1, pts2)
    return cv2.warpAffine(img, M, (w, h))


def crop(img: np.ndarray):
    """
    Crop image by 10%
    """
    h, w = img.shape[:2]
    y1 = int(0.1 * h)
    y2 = int(0.9 * h)
    x1 = int(0.1 * w)
    x2 = int(0.9 * w)
    return img[y1:y2, x1:x2]


def distortion(img: np.ndarray):
    """
    Blur the image
    """
    return cv2.GaussianBlur(img, (7, 7), 0)


def contrast(img: np.ndarray):
    """
    alpha = contrast factor
    beta = brightness
    """
    alpha = 1.5
    beta = 20
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def augmentation(filepath: Path) -> None:
    """
    Perform six different augmentation on the image filepath
    and save to the same directory.
    """
    current_dir = filepath.parent
    filename = filepath.stem

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
        save_path = current_dir / f"{filename}_{aug_name}.JPG"
        cv2.imwrite(str(save_path), aug_img)


def main():
    """main()"""

    try:

        if len(sys.argv) != 2:
            print("Error: the arguments are bad")
            return

        path = Path(sys.argv[1])

        if path.is_dir():
            augmented_suffixes = (
                "_Flip", "_Rotate", "_Skew",
                "_Contrast", "_Crop", "_Distortion"
            )
            images = [
                f for f in path.rglob("*.JPG")
                if not any(f.stem.endswith(s) for s in augmented_suffixes)
            ]
            for img_path in sorted(images):
                print(f"Augmenting {img_path.name}...")
                augmentation(img_path)
        else:
            is_image_file(path)
            augmentation(path)

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
