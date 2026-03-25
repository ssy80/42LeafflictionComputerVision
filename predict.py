import sys
import numpy as np
from pathlib import Path
from utils import is_image_file, is_path_dir
from Transformation import transformation
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib


def main():
    """main()"""

    try:
 
        if len(sys.argv) != 2:
            print("Error: the arguments are bad")
            return

        filepath = Path(sys.argv[1])
        is_image_file(filepath)

        # 1. Load the "frozen" tools
        model = joblib.load('./rf_leaf_classifier.pkl')
        encoder = joblib.load('le.pkl')

        # 2. Extract features from a NEW image (using your transformation function)
        _, predict_features_dict = transformation(filepath)
        df = pd.DataFrame([predict_features_dict])

        # 3. Predict!
        prediction_numeric = model.predict(df)
        prediction_text = encoder.inverse_transform(prediction_numeric)

        print(f"The model identifies this leaf as: {prediction_text[0]}")

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
