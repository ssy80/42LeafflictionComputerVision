import numpy as np
from augmentation.transforms import flip, rotate, skew, crop, distortion
from augmentation.augmentation import contrast, augmentation


def test_flip_returns_ndarray(sample_bgr_image):
    result = flip(sample_bgr_image)
    assert isinstance(result, np.ndarray)
    assert result.shape == sample_bgr_image.shape


def test_rotate_returns_ndarray(sample_bgr_image):
    result = rotate(sample_bgr_image)
    assert isinstance(result, np.ndarray)
    assert result.ndim == 3


def test_skew_preserves_shape(sample_bgr_image):
    result = skew(sample_bgr_image)
    assert result.shape == sample_bgr_image.shape


def test_crop_returns_smaller_image(sample_bgr_image):
    result = crop(sample_bgr_image)
    h, w = sample_bgr_image.shape[:2]
    assert result.shape[0] < h
    assert result.shape[1] < w


def test_distortion_preserves_shape(sample_bgr_image):
    result = distortion(sample_bgr_image)
    assert result.shape == sample_bgr_image.shape


def test_contrast_preserves_shape(sample_bgr_image):
    result = contrast(sample_bgr_image)
    assert result.shape == sample_bgr_image.shape


def test_flip_is_mirrored(sample_bgr_image):
    result = flip(sample_bgr_image)
    assert np.array_equal(sample_bgr_image[:, 0], result[:, -1])


def test_augmentation_saves_six_variants(sample_image_file):
    augmentation(sample_image_file)
    saved = list(sample_image_file.parent.glob("*.JPG"))
    # original + 6 augmented variants
    assert len(saved) == 7
