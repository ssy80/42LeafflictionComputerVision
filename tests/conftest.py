import pytest
import numpy as np
import cv2


@pytest.fixture
def sample_bgr_image():
    """256x256 BGR image with a green leaf-like centre on a white background."""
    img = np.full((256, 256, 3), 255, dtype=np.uint8)
    img[64:192, 64:192] = [34, 139, 34]  # forest green in BGR
    return img


@pytest.fixture
def sample_image_file(tmp_path, sample_bgr_image):
    """Write the sample image to a temporary .JPG file and return its Path."""
    img_path = tmp_path / "leaf.JPG"
    cv2.imwrite(str(img_path), sample_bgr_image)
    return img_path


@pytest.fixture
def sample_image_dir(tmp_path, sample_bgr_image):
    """Temporary directory containing three .JPG files."""
    for i in range(3):
        cv2.imwrite(str(tmp_path / f"image_{i}.JPG"), sample_bgr_image)
    return tmp_path


@pytest.fixture
def sample_class_dir(tmp_path, sample_bgr_image):
    """
    Dataset root with two class subdirectories (class_a, class_b),
    each containing four .JPG images — ready for split_dataset.
    """
    for cls in ("class_a", "class_b"):
        cls_dir = tmp_path / cls
        cls_dir.mkdir()
        for i in range(4):
            cv2.imwrite(str(cls_dir / f"image_{i}.JPG"), sample_bgr_image)
    return tmp_path
