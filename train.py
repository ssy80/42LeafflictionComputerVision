import sys
#import os
import numpy as np
from pathlib import Path
from utils import is_image_file, is_path_dir
from Transformation import transformation
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
import joblib

def train(df: pd.DataFrame):
    """
    """
    y = df["label"]
    X = df.drop(columns=["image", "label"])#, "disease_area", "longest_path"])
    
    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)

    joblib.dump(rf, 'rf_leaf_classifier.pkl')
    joblib.dump(le, 'le.pkl')

    scores = cross_val_score(rf, X, y, cv=5)
    print(f"Mean Accuracy: {scores.mean():.2%}")

    y_pred = rf.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")


def get_dataset(dirpath: str):
    """
    """
    base_path = Path(dirpath)
    dataset = []

    for subdir in base_path.iterdir():
        if subdir.is_dir():
            label = subdir.name
            
            for file_path in subdir.glob("*.JPG"): 
                _, result_dict = transformation(file_path)
                result_dict["image"] = file_path.name
                result_dict["label"] = label

                dataset.append(result_dict)
    
    df = pd.DataFrame(dataset)
    df.to_csv("./leaf_features.csv", index=False)
    print(df)
    return df


def main():
    """main()"""

    try:
 
        if len(sys.argv) != 2:
            print("Error: the arguments are bad")
            return

        dirpath = Path(sys.argv[1])
        is_path_dir(dirpath)

        #df = get_dataset(dirpath)
        df = pd.read_csv("./leaf_features.csv")
        train(df)

    except Exception as e:
        print(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
