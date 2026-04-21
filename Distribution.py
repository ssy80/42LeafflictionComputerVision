import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path
from utils import is_path_dir


def list_dirs(dir_path: str)-> list[str]:
    """
    Return a sorted list of directories inside ``path``.
    """
    is_path_dir(dir_path)    
    base_path = Path(dir_path)

    directories = []
    directories = [str(entry) for entry in base_path.iterdir() if entry.is_dir()]

    return sorted(directories)


def count_images(dir_path: str)-> int:
    """
    Count number of image file in the dir_path
    """
    is_path_dir(dir_path)
    path = Path(dir_path)

    image_extensions = {".jpg", ".jpeg"}

    return sum(
        1
        for entry in path.iterdir()
        if entry.is_file() and entry.suffix.lower() in image_extensions
    )


def pie_chart(df: pd.DataFrame, dir_name: str)-> None:
    """
    Display a pie chart of the leave distribution
    """
    plt.pie(
        df["count"],
        labels=df.index,
        autopct="%1.1f%%",
        startangle=180,
    )
    plt.title(f"{dir_name} class Distribution")
    plt.show()
    

def bar_chart(df: pd.DataFrame, dir_name: str)-> None:
    """
    Display a bar chart with class on x-axis and counts on y-axis.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(x=df.index, y=df["count"], palette="viridis", hue=df.index, legend=False)
    plt.title(f"{dir_name} class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def main():
    """main()"""

    try:
 
        if len(sys.argv) != 2:
            print("Error: the arguments are bad")
            return
    
        dir_path = str(sys.argv[1])
        dir_name = Path(dir_path).name
        sub_dirs = list_dirs(dir_path)

        images_in_dir_dict = {}
        for leaf_dir in sub_dirs:
            count = count_images(leaf_dir)
            folder_name = Path(leaf_dir).name
            images_in_dir_dict[folder_name] = count

        df = pd.DataFrame.from_dict(images_in_dir_dict, orient='index', columns=['count'])
        
        pie_chart(df, dir_name)
        bar_chart(df, dir_name)

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
