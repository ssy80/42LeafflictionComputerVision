import sys
import numpy as np
import pandas as pd
from pathlib import Path
from utils import is_image_file, is_path_dir
import tensorflow as tf
import matplotlib.pyplot as plt


def main():
    """main()"""

    try:

        if len(sys.argv) != 4:
            print("Error: the arguments are bad")
            return

        filepath = Path(sys.argv[1])
        is_image_file(filepath)

        model_path = Path(sys.argv[2])
        is_path_dir(model_path)

        original_path = Path(sys.argv[3])
        is_path_dir(original_path)

        loaded_model = tf.keras.models.load_model(
            model_path / "leaf_model.keras"
            )

        df = pd.read_csv(model_path / "class_names.csv")
        class_names = df["class_name"].tolist()

        img = tf.keras.utils.load_img(filepath, target_size=(256, 256))
        img_array = tf.keras.utils.img_to_array(img)

        # add batch dimension so shape becomes (1, 256, 256, 3)
        img_array = np.expand_dims(img_array, axis=0)

        pred = loaded_model.predict(img_array)

        predicted_class = np.argmax(pred, axis=1)[0]
        predicted_class_label = class_names[predicted_class]

        print("Predicted class index:", predicted_class)
        print("Predicted class label:", class_names[predicted_class])

        predict_image_name = filepath.stem
        splited_base_image_name = predict_image_name.split("_")[0]
        original_image_path = original_path /\
            predicted_class_label / f"{splited_base_image_name}.JPG"

        img_original = tf.keras.utils.load_img(
            original_image_path, target_size=(256, 256)
            )

        fig = plt.figure(figsize=(8, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(img_original)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(img)
        plt.title("Predict")
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
