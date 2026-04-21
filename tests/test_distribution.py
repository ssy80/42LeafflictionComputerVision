import pytest
from pathlib import Path
from Distribution import count_images, list_dirs


def test_count_images_correct(sample_image_dir):
    assert count_images(str(sample_image_dir)) == 3


def test_count_images_empty_dir(tmp_path):
    assert count_images(str(tmp_path)) == 0


def test_count_images_ignores_non_jpg(tmp_path):
    (tmp_path / "file.png").write_text("x")
    (tmp_path / "file.txt").write_text("x")
    assert count_images(str(tmp_path)) == 0


def test_list_dirs_returns_sorted(tmp_path):
    for name in ("zebra", "apple", "mango"):
        (tmp_path / name).mkdir()
    result = list_dirs(str(tmp_path))
    assert result == sorted(result)


def test_list_dirs_excludes_files(tmp_path):
    (tmp_path / "not_a_dir.txt").write_text("x")
    (tmp_path / "subdir").mkdir()
    result = list_dirs(str(tmp_path))
    assert all(Path(p).is_dir() for p in result)


def test_list_dirs_invalid_path():
    with pytest.raises(FileNotFoundError):
        list_dirs("/nonexistent/path")
