from pathlib import Path


def is_path_dir(dir_path: str)-> None:
    """
    Check is path a valid directory
    """
    base_path = Path(dir_path)

    if not base_path.exists():
        raise FileNotFoundError(f"Path does not exist: {dir_path}")

    if not base_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {dir_path}")
