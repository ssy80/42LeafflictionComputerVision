import numpy as np
from Transformation import original, gaussian_blur, mask, roi, transformation, transform_dir


def test_original_returns_same(sample_bgr_image):
    result = original(sample_bgr_image.copy())
    assert np.array_equal(result, sample_bgr_image)


def test_gaussian_blur_returns_2d(sample_bgr_image):
    result = gaussian_blur(sample_bgr_image.copy())
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2


def test_mask_returns_3channel(sample_bgr_image):
    result = mask(sample_bgr_image.copy())
    assert isinstance(result, np.ndarray)
    assert result.ndim == 3
    assert result.shape[2] == 3


def test_roi_preserves_spatial_dims(sample_bgr_image):
    result = roi(sample_bgr_image.copy())
    assert isinstance(result, np.ndarray)
    assert result.shape[:2] == sample_bgr_image.shape[:2]


def test_transformation_returns_all_keys(sample_image_file):
    expected = {"original", "gaussian_blur", "mask", "roi", "analyze", "pseudolandmarks"}
    result = transformation(sample_image_file)
    assert set(result.keys()) == expected


def test_transformation_all_values_are_ndarrays(sample_image_file):
    result = transformation(sample_image_file)
    for key, val in result.items():
        assert isinstance(val, np.ndarray), f"{key} did not return an ndarray"


def test_transform_dir_writes_six_files(tmp_path, sample_image_file):
    src = sample_image_file.parent
    dst = tmp_path / "out"
    dst.mkdir()
    transform_dir(src, dst)
    outputs = list(dst.glob("*.JPG"))
    # 1 source image × 6 transformations = 6 output files
    assert len(outputs) == 6
