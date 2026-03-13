import cv2
import sys
import os
import numpy as np
from pathlib import Path


def is_image_file(filepath: Path)-> None:
    """
    Check filepath is an image file (.jpg, .jpeg) 
    """
    if not filepath.is_file():
        raise FileNotFoundError(f"file not found: {filepath}")

    if filepath.suffix.lower() not in (".jpg", ".jpeg"):
        raise ValueError("file must be a .jpg or .jpeg image")


def flip(img: np.ndarray):
    """
    Flip image to the right
    """
    return cv2.flip(img, 1)


def rotate(img: np.ndarray):
    """
    Rotate to right 25 degree angle.
    cv2.getRotationMatrix2D(center, angle, scale)
    """
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), -25, 1)
    return cv2.warpAffine(img, M, (w, h))


def skew(img: np.ndarray):
    """
    Skew image
    """
    h, w = img.shape[:2]
    pts1 = np.float32([[0,0],[w,0],[0,h]])
    pts2 = np.float32([[0,0],[w,0],[int(0.2*w),h]])
    M = cv2.getAffineTransform(pts1,pts2)
    return cv2.warpAffine(img,M,(w,h))


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
    return cv2.GaussianBlur(img,(7,7),0)


def contrast(img: np.ndarray):
    """
    alpha = contrast factor
    beta = brightness
    """
    alpha = 1.5
    beta = 20
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def  augmentation(filepath: Path)-> None:
    """
    Perform six different augmentation on the image filepath
    and save to the same directory.
    """
    current_dir = filepath.parent
    print(current_dir)

    filename = filepath.stem
    print(filename)

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

        filepath = Path(sys.argv[1])
        print(filepath)

        is_image_file(filepath)
        augmentation(filepath)

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
