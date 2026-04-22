import cv2
import numpy as np
from pathlib import Path
import argparse
from plantcv import plantcv as pcv
import matplotlib.pyplot as plt
from transformation.filters import gaussian_blur, mask, roi, analyze, pseudolandmarks
from utils.utils import is_path_dir, is_image_file


def original(img: np.ndarray):
    """
    return original image
    """
    return img


def plot_leaf_color_histogram(img, mask=None):
    """
    Plot normalized histograms of multiple color channels from a leaf image.
    This function optionally applies a binary mask to keep only
    selected pixels, converts the image into RGB, HSV, and LAB
    color spaces, extracts the channel values from the selected region,
    and plots each channel histogram as the percentage of valid pixels
    at each intensity level.
    """
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    if mask is None:
        mask = np.full(img.shape[:2], 255, dtype="uint8")

    valid = mask > 0

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    red = rgb[:, :, 0][valid]
    green = rgb[:, :, 1][valid]
    blue = rgb[:, :, 2][valid]

    hue = hsv[:, :, 0][valid]
    saturation = hsv[:, :, 1][valid]
    value = hsv[:, :, 2][valid]

    lightness = lab[:, :, 0][valid]
    green_magenta = lab[:, :, 1][valid]
    blue_yellow = lab[:, :, 2][valid]

    channels = {
        "blue": blue,
        "blue-yellow": blue_yellow,
        "green": green,
        "green-magenta": green_magenta,
        "hue": hue,
        "lightness": lightness,
        "red": red,
        "saturation": saturation,
        "value": value,
    }

    colors = {
        "blue": "blue",
        "blue-yellow": "yellow",
        "green": "green",
        "green-magenta": "magenta",
        "hue": "#7d3cff",
        "lightness": "gray",
        "red": "red",
        "saturation": "cyan",
        "value": "orange",
    }

    plt.figure(figsize=(11, 6))

    total_pixels = np.count_nonzero(valid)

    for name, vals in channels.items():
        hist, bins = np.histogram(vals, bins=256, range=(0, 256))
        hist_percent = (hist / total_pixels) * 100
        plt.plot(
            bins[:-1],
            hist_percent,
            label=name,
            color=colors[name],
            linewidth=1.5
            )

    plt.xlabel("Pixel intensity", fontsize=16)
    plt.ylabel("Proportion of pixels (%)", fontsize=16)
    plt.xlim(0, 255)
    plt.legend(
        title="color Channel",
        bbox_to_anchor=(1.03, 0.5),
        loc="center left"
        )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def transformation(filepath: Path) -> None:
    """
    Various transformation to a single image
    Extract features from various transformations
    """
    img = cv2.imread(str(filepath))

    transformations = {
        "original": original(img.copy()),
        "gaussian_blur": gaussian_blur(img.copy()),
        "mask": mask(img.copy()),
        "roi": roi(img.copy()),
        "analyze": analyze(img.copy()),
        "pseudolandmarks": pseudolandmarks(img.copy())
    }

    return transformations


def transform_dir(src_path: Path, dest_path: Path) -> None:
    """
    Loop the src directory
    Transform every image in the src directory
    Save to dst directory
    """
    dest_path.mkdir(parents=True, exist_ok=True)

    for file_path in src_path.glob("*.JPG"):
        filename = file_path.stem
        transformed = transformation(file_path)
        for trn_name, trn_img in transformed.items():
            save_path = dest_path / f"{filename}_{trn_name}.JPG"
            cv2.imwrite(str(save_path), trn_img)


def main():
    """main()"""

    try:

        parser = argparse.ArgumentParser()
        parser.add_argument("-src", required=True)
        parser.add_argument("-dst", required=False)

        args = parser.parse_args()
        src_path = Path(args.src)

        dest_path = None
        if args.dst:
            dest_path = Path(args.dst)

        if dest_path:
            is_path_dir(src_path)
            transform_dir(src_path, dest_path)
        else:
            is_image_file(src_path)
            transformed = transformation(src_path)
            for _, trn_img in transformed.items():
                pcv.plot_image(trn_img)
                plot_leaf_color_histogram(trn_img)

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
