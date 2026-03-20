import argparse
from pathlib import Path

import cv2
import numpy as np
from plantcv import plantcv as pcv

from utils import is_path_dir


LEAF_MASK_SUFFIX = "_leaf_mask.png"
STRUCTURE_MASK_SUFFIX = "_structure_mask.png"
TRANSFORM_SUFFIX = "_mask_transform.png"


def is_image_file(filepath: Path) -> None:
    """Validate that the source path points to a supported image file."""
    if not filepath.is_file():
        raise FileNotFoundError(f"file not found: {filepath}")

    if filepath.suffix.lower() not in (".jpg", ".jpeg", ".png"):
        raise ValueError("file must be a .jpg, .jpeg, or .png image")


def keep_largest_contour(mask: np.ndarray) -> np.ndarray:
    """Reduce a binary mask to its largest connected contour."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = np.zeros_like(mask)

    if contours:
        cv2.drawContours(largest, [max(contours, key=cv2.contourArea)], -1, 255, thickness=cv2.FILLED)

    return largest


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Remove connected components smaller than the requested area."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    filtered = np.zeros_like(mask)

    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            filtered[labels == label] = 255

    return filtered


def build_leaf_mask(img: np.ndarray) -> np.ndarray:
    """Create a clean leaf silhouette from the original RGB image."""
    saturation = pcv.rgb2gray_hsv(rgb_img=img, channel="s")
    saturation = pcv.gaussian_blur(img=saturation, ksize=(5, 5), sigma_x=0, sigma_y=None)
    saturation_mask = pcv.threshold.binary(gray_img=saturation, threshold=35, object_type="light")

    green_magenta = pcv.rgb2gray_lab(rgb_img=img, channel="a")
    green_mask = pcv.threshold.binary(gray_img=green_magenta, threshold=121, object_type="dark")

    leaf_mask = cv2.bitwise_or(saturation_mask, green_mask)
    leaf_mask = pcv.fill_holes(leaf_mask)
    leaf_mask = remove_small_components(leaf_mask, min_area=300)

    return keep_largest_contour(leaf_mask)


def build_structure_mask(img: np.ndarray, leaf_mask: np.ndarray) -> np.ndarray:
    """Keep the stronger veins, edges, and lesions inside the leaf mask."""
    hue = pcv.rgb2gray_hsv(rgb_img=img, channel="h")
    saturation = pcv.rgb2gray_hsv(rgb_img=img, channel="s")
    value = pcv.rgb2gray_hsv(rgb_img=img, channel="v")
    blue_yellow = pcv.rgb2gray_lab(rgb_img=img, channel="b")

    green_hue_mask = cv2.inRange(hue, 28, 95)
    saturated_mask = pcv.threshold.binary(gray_img=saturation, threshold=55, object_type="light")
    darker_green_mask = pcv.threshold.binary(gray_img=value, threshold=175, object_type="dark")
    green_structure = cv2.bitwise_and(green_hue_mask, saturated_mask)
    green_structure = cv2.bitwise_and(green_structure, darker_green_mask)

    lesion_mask = pcv.threshold.binary(gray_img=blue_yellow, threshold=130, object_type="dark")

    edge_map = pcv.laplace_filter(gray_img=value, ksize=3, scale=1)
    edge_map = cv2.convertScaleAbs(edge_map)
    edge_map = pcv.gaussian_blur(img=edge_map, ksize=(3, 3), sigma_x=0, sigma_y=None)
    edge_mask = pcv.threshold.binary(gray_img=edge_map, threshold=35, object_type="light")
    edge_mask = pcv.dilate(gray_img=edge_mask, ksize=3, i=1)

    structure_mask = cv2.bitwise_or(green_structure, lesion_mask)
    structure_mask = cv2.bitwise_or(structure_mask, edge_mask)
    structure_mask = cv2.bitwise_and(structure_mask, leaf_mask)
    structure_mask = cv2.morphologyEx(structure_mask, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))

    return structure_mask


def render_mask_transform(
    img: np.ndarray, leaf_mask: np.ndarray, structure_mask: np.ndarray
) -> np.ndarray:
    """Render a white background result that preserves only masked leaf details."""
    result = np.full_like(img, 255)
    result[structure_mask > 0] = img[structure_mask > 0]

    contours, _ = cv2.findContours(leaf_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(result, contours, -1, (0, 0, 0), thickness=2)

    return result


def transformation(filepath: Path, dirpath: Path) -> None:
    """Build and save PlantCV-driven mask outputs for one plant image."""
    img = cv2.imread(str(filepath))
    if img is None:
        raise ValueError(f"unable to read image: {filepath}")

    filename = filepath.stem
    leaf_mask = build_leaf_mask(img)
    structure_mask = build_structure_mask(img, leaf_mask)
    transformed = render_mask_transform(img, leaf_mask, structure_mask)

    cv2.imwrite(str(dirpath / f"{filename}{LEAF_MASK_SUFFIX}"), leaf_mask)
    cv2.imwrite(str(dirpath / f"{filename}{STRUCTURE_MASK_SUFFIX}"), structure_mask)
    cv2.imwrite(str(dirpath / f"{filename}{TRANSFORM_SUFFIX}"), transformed)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate PlantCV leaf masks and a mask-style transformed image."
    )
    parser.add_argument("-src", required=True, help="source image path")
    parser.add_argument("-dst", required=True, help="destination directory")

    args = parser.parse_args()

    filepath = Path(args.src)
    dirpath = Path(args.dst)

    is_path_dir(dirpath)
    is_image_file(filepath)
    transformation(filepath, dirpath)


if __name__ == "__main__":
    main()
