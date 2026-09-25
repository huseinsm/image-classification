import argparse
import os
import pickle

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC

from features import CATEGORIES, image_to_features


def load_dataset(data_dir):
    data, labels = [], []
    for label, category in enumerate(CATEGORIES):
        folder = os.path.join(data_dir, category)
        for name in sorted(os.listdir(folder)):
            data.append(image_to_features(os.path.join(folder, name)))
            labels.append(label)
    return np.asarray(data), np.asarray(labels)


def main():
    parser = argparse.ArgumentParser(description="Train an SVM that tells empty parking spots from occupied ones.")
    parser.add_argument("--data", default="clf-data", help="folder containing empty/ and not_empty/")
    parser.add_argument("--model", default="model.p", help="where to save the trained classifier")
    args = parser.parse_args()

    X, y = load_dataset(args.data)
    print(f"loaded {len(X)} images ({np.bincount(y).tolist()} per class)")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42
    )

    grid = GridSearchCV(SVC(), [{"gamma": [0.01, 0.001, 0.0001], "C": [1, 10, 100, 1000]}])
    grid.fit(X_train, y_train)
    model = grid.best_estimator_

    y_pred = model.predict(X_test)
    print("best params:", grid.best_params_)
    print(classification_report(y_test, y_pred, target_names=CATEGORIES))
    print("confusion matrix:\n", confusion_matrix(y_test, y_pred))

    with open(args.model, "wb") as f:
        pickle.dump(model, f)
    print(f"model saved -> {args.model}")


if __name__ == "__main__":
    main()
