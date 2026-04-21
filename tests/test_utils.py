import pytest
from pathlib import Path
from utils import is_path_dir, is_image_file


def test_is_path_dir_valid(tmp_path):
    is_path_dir(tmp_path)  # should not raise


def test_is_path_dir_missing():
    with pytest.raises(FileNotFoundError):
        is_path_dir(Path("/nonexistent/path"))


def test_is_path_dir_not_a_dir(tmp_path):
    f = tmp_path / "file.txt"
    f.write_text("x")
    with pytest.raises(NotADirectoryError):
        is_path_dir(f)


def test_is_image_file_valid(sample_image_file):
    is_image_file(sample_image_file)  # should not raise


def test_is_image_file_missing():
    with pytest.raises(FileNotFoundError):
        is_image_file(Path("/nonexistent/image.JPG"))


def test_is_image_file_wrong_extension(tmp_path):
    f = tmp_path / "file.png"
    f.write_text("x")
    with pytest.raises(ValueError):
        is_image_file(f)
