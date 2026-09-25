import argparse
import pickle

from features import CATEGORIES, image_to_features


def main():
    parser = argparse.ArgumentParser(description="Classify parking-spot images as empty or not_empty.")
    parser.add_argument("images", nargs="+", help="one or more image paths")
    parser.add_argument("--model", default="model.p")
    args = parser.parse_args()

    with open(args.model, "rb") as f:
        model = pickle.load(f)

    predictions = model.predict([image_to_features(p) for p in args.images])
    for path, label in zip(args.images, predictions):
        print(f"{CATEGORIES[label]:>9}  {path}")


if __name__ == "__main__":
    main()
