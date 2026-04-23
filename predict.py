#!/usr/bin/env my_env/bin/python3
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from utils import is_image_file, is_path_dir
import tensorflow as tf
import matplotlib.pyplot as plt
from Transformation import transformation


def main():
    """main()"""

    try:

        if len(sys.argv) not in (2, 3):
            print("Error: the arguments are bad")
            return

        to_predict_src = Path(sys.argv[1])
        is_image_file(to_predict_src)

        if len(sys.argv) == 3:
            model_path = Path(sys.argv[2])
        else:
            all_parts = list(to_predict_src.parts) + [to_predict_src.stem]
            model_path = None
            for plant in ("Apple", "Grape"):
                if any(plant in part for part in all_parts):
                    candidates = [
                        Path("augmented_directory") / plant / "splited",
                        Path("test_augmented") / plant / "splited",
                        Path("test") / plant / "splited",
                        Path("models") / plant / "splited",
                    ]
                    for candidate in candidates:
                        if candidate.exists():
                            model_path = candidate
                            break
                    break
            if model_path is None:
                print("Error: cannot auto-detect model. "
                      "Provide model path as second argument.")
                return
        is_path_dir(model_path)

        transformations = transformation(to_predict_src)
        img_transformed = transformations["mask"]

        loaded_model = tf.keras.models.load_model(
            model_path / "leaf_model.keras"
            )

        df = pd.read_csv(model_path / "class_names.csv")
        class_names = df["class_name"].tolist()

        img = tf.keras.utils.load_img(to_predict_src, target_size=(128, 128))
        img_array = tf.keras.utils.img_to_array(img)

        # add batch dimension so shape becomes (1, 256, 256, 3)
        img_array = np.expand_dims(img_array, axis=0)

        pred = loaded_model.predict(img_array)

        predicted_class = np.argmax(pred, axis=1)[0]

        print("Predicted class index:", predicted_class)
        print("Predicted class label:", class_names[predicted_class])

        fig = plt.figure(figsize=(8, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(img_transformed)
        plt.title("Transformed")
        plt.axis("off")

        fig.text(
            0.5, 0.02,
            f"Predicted class: {class_names[predicted_class]}",
            ha="center",
            fontsize=12
        )

        plt.show()

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
