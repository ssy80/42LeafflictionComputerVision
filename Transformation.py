import cv2
import sys
import os
import numpy as np
from pathlib import Path
import argparse


def is_image_file(filepath: Path)-> None:
    """
    Check filepath is an image file (.jpg, .jpeg) 
    """
    if not filepath.is_file():
        raise FileNotFoundError(f"file not found: {filepath}")

    if filepath.suffix.lower() not in (".jpg", ".jpeg"):
        raise ValueError("file must be a .jpg or .jpeg image")


def  transformation(filepath: Path)-> None:
    """
    Perform six different  on the image filepath
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
 
        '''
        if len(sys.argv) != 2:
            print("Error: the arguments are bad")
            return

        filepath = Path(sys.argv[1])
        print(filepath)

        is_image_file(filepath)
        transformation(filepath)'''
        parser = argparse.ArgumentParser()

        parser.add_argument("-src", required=True)
        parser.add_argument("-dst", required=True)
        parser.add_argument("-mask", action="store_true")

        args = parser.parse_args()

        print(args.src)
        print(args.dst)
        print(args.mask)

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
