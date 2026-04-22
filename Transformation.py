#!/usr/bin/env my_env/bin/python3
import cv2
import os
import numpy as np
from pathlib import Path
import argparse
from utils import is_path_dir, is_image_file
from plantcv import plantcv as pcv
import glob
import matplotlib.pyplot as plt
import shutil


def plot_leaf_color_histogram(img, mask=None):
    """
    Plot normalized histograms of multiple color channels from a leaf image.
    This function optionally applies a binary mask to keep only
    selected pixels, converts the image into RGB, HSV, and LAB
    color spaces, extracts the channel values from the selected region,
    and plots each channel histogram as the percentage of valid pixels
    at each intensity level.
    """
    # convert to ensure 3-channel image
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    if mask is None:
        mask = np.full(img.shape[:2], 255, dtype="uint8")
    elif len(mask.shape) == 3:
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY_INV)

    valid = mask > 0

    # Convert color spaces
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # Extract channels inside mask
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


def pseudolandmarks(img: np.ndarray):
    """
    Extract saturation (s) channel
    Gaussian blur the image
    Threshold saturation using otsu - auto threshold
    Remove small objects
    Remove salt-pepper noise
    Use pseudolandmarks y-axis
    Return the image created
    """
    # low saturation -> dull / grayish
    # high saturation -> rich, vivid, strong color
    s = pcv.rgb2gray_hsv(rgb_img=img, channel='s')

    blur = pcv.gaussian_blur(img=s, ksize=(5, 5), sigma_x=0, sigma_y=None)
    mask = pcv.threshold.otsu(blur, object_type='light')
    # mask = pcv.fill_holes(mask)
    mask = pcv.fill(mask, size=50)
    mask = pcv.median_blur(mask, ksize=3)

    pcv.params.text_size = 0
    pcv.params.debug = "print"
    pcv.params.debug_outdir = "./debug"
    debug_dir = pcv.params.debug_outdir

    if os.path.exists(debug_dir):
        shutil.rmtree(debug_dir)
        os.makedirs(debug_dir)
    else:
        os.makedirs(debug_dir)

    pcv.outputs.clear()

    pcv.homology.y_axis_pseudolandmarks(img=img, mask=mask)

    files = glob.glob(os.path.join(debug_dir, "*"))
    final_img = cv2.imread(files[0])

    return final_img


def analyze(img: np.ndarray):
    """
    Extract saturation (s) channel
    Gaussian blur the image
    Threshold saturation using otsu - auto threshold
    Remove small objects
    Remove salt-pepper noise
    use plantcv analyze
    """
    s = pcv.rgb2gray_hsv(rgb_img=img, channel='s')

    blur = pcv.gaussian_blur(img=s, ksize=(5, 5), sigma_x=0, sigma_y=None)
    mask = pcv.threshold.otsu(blur, object_type='light')
    # mask = pcv.fill_holes(mask)
    mask = pcv.fill(mask, size=50)
    mask = pcv.median_blur(mask, ksize=3)

    pcv.params.text_size = 0
    pcv.outputs.clear()

    analyze_img = pcv.analyze.size(
        img=img,
        labeled_mask=mask,
        n_labels=1,
        label="leaf_data"
        )

    return analyze_img


def roi(img: np.ndarray):
    """
    Get HSV from BGR
    Extract saturation (s) channel
    Define green range
    Get healthy mask from green range
    Gaussian blur the image
    Threshold saturation using otsu - auto threshold
    Remove small objects
    Remove salt-pepper noise
    Get final healthy mask from matching threshold mask
    Overlay final healthy mask to the img
    Draw blue rectangle
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]

    lower_green = (25, 25, 30)
    upper_green = (95, 255, 255)

    healthy_mask = cv2.inRange(hsv, lower_green, upper_green)

    blur = pcv.gaussian_blur(img=s, ksize=(5, 5), sigma_x=0, sigma_y=None)
    mask = pcv.threshold.otsu(blur, object_type='light')
    # mask = pcv.fill_holes(mask)
    mask = pcv.fill(mask, size=50)
    mask = pcv.median_blur(mask, ksize=3)

    # Only keep Green pixels that are actually INSIDE the leaf
    final_healthy_mask = cv2.bitwise_and(healthy_mask, mask)

    overlay = img.copy()
    overlay[final_healthy_mask > 0] = (0, 255, 0)

    h, w = overlay.shape[:2]

    # Draw a blue rectangle around the entire image
    cv2.rectangle(
        overlay,
        (0, 0),
        (w-1, h-1),
        (255, 0, 0),
        3
    )

    return overlay


def mask(img: np.ndarray):
    """
    Extract saturation (s) channel
    Gaussian blur the image
    Threshold saturation using otsu - auto threshold
    Remove small objects
    Remove salt-pepper noise
    Apply mask to image
    """
    s = pcv.rgb2gray_hsv(rgb_img=img, channel='s')

    blur = pcv.gaussian_blur(img=s, ksize=(5, 5), sigma_x=0, sigma_y=None)
    mask = pcv.threshold.otsu(blur, object_type='light')

    # mask = pcv.fill_holes(mask)
    mask = pcv.fill(mask, size=50)
    mask = pcv.median_blur(mask, ksize=3)

    masked_img = pcv.apply_mask(
        img=img,
        mask=mask,
        mask_color='white'
    )

    return masked_img


def gaussian_blur(img: np.ndarray):
    """
    Extract saturation (s) channel
    Gaussian blur the image
    Threshold saturation using otsu - auto threshold
    """
    s = pcv.rgb2gray_hsv(rgb_img=img, channel='s')

    blur = pcv.gaussian_blur(img=s, ksize=(5, 5), sigma_x=0, sigma_y=None)

    blur_img = pcv.threshold.otsu(blur, object_type='light')

    return blur_img


def original(img: np.ndarray):
    """
    return original image
    """
    return img


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


def display_transformations(transformed: dict) -> None:
    titles = {
        "original": "Original",
        "gaussian_blur": "Gaussian blur",
        "mask": "Mask",
        "roi": "Roi objects",
        "analyze": "Analyze object",
        "pseudolandmarks": "Pseudolandmarks",
    }
    fig, axes = plt.subplots(3, 2, figsize=(10, 12))
    for i, (ax, (key, img)) in enumerate(zip(axes.flat, transformed.items())):
        if len(img.shape) == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_xlabel(f"Figure IV.{i + 1}: {titles.get(key, key)}", fontsize=11)
        ax.xaxis.set_label_position("bottom")
    plt.tight_layout()
    plt.show()


def main():
    """main()"""

    try:

        import sys as _sys
        if len(_sys.argv) > 1 and not _sys.argv[1].startswith('-'):
            src_path = Path(_sys.argv[1])
            is_image_file(src_path)
            img = cv2.imread(str(src_path))
            transformed = transformation(src_path)
            display_transformations(transformed)
            plot_leaf_color_histogram(img, transformed.get("mask"))
            return

        parser = argparse.ArgumentParser(
            description="Apply 6 image transformations to a leaf image.",
            usage="%(prog)s <image> | %(prog)s -src <path> [-dst <dir>] [-mask]"
        )
        parser.add_argument("-src", required=True, help="Source image or directory")
        parser.add_argument("-dst", required=False, help="Destination directory (batch mode)")
        parser.add_argument("-mask", action="store_true", help="Apply mask filter in batch mode")

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
            img = cv2.imread(str(src_path))
            transformed = transformation(src_path)
            display_transformations(transformed)
            plot_leaf_color_histogram(img, transformed.get("mask"))

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
