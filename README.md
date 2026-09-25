# Image Classification: Empty vs. Occupied Parking Spots

A small, fast image classifier that looks at a crop of a single parking spot and says whether it's **empty** or **not_empty**. It is the classification half of a parking-lot occupancy counter: slice the camera frame into spots, classify each one, and count.

No deep learning is needed. The images are resized to 15×15 pixels, flattened, and passed to a Support Vector Machine. Inference is fast enough to classify every spot in a lot on every frame.

## Results

Trained on 6,090 images (3,045 per class) with a stratified 80/20 split:

| | Precision | Recall | Support |
|---|---|---|---|
| empty | 1.00 | 1.00 | 609 |
| not_empty | 1.00 | 1.00 | 609 |

**1,218 / 1,218 test images correct.** Best hyperparameters from a 5-fold grid search: `C=10, gamma=0.01` (RBF kernel).

Take that number with some salt. All crops come from the same camera and lot, and consecutive frames are nearly identical, so near-duplicates of a test image are probably in the training set. The model will be less accurate on a different parking lot or under different lighting.

## How it works

`features.py` resizes each image to 15×15 RGB and flattens it into a 675-value vector. `train.py` then:

1. loads every image in `empty/` and `not_empty/`
2. makes a stratified train/test split (`random_state=42`)
3. grid-searches an SVC over `C ∈ {1, 10, 100, 1000}` and `gamma ∈ {0.01, 0.001, 0.0001}`
4. prints a classification report and confusion matrix
5. pickles the best model

## Usage

```bash
pip install -r requirements.txt

# train (about 3 minutes on a laptop CPU)
python train.py --data path/to/clf-data --model model.p

# classify one or more crops
python predict.py --model model.p spot1.jpg spot2.jpg
#     empty  spot1.jpg
# not_empty  spot2.jpg
```

The data folder should look like this:

```
clf-data/
  empty/       *.jpg
  not_empty/   *.jpg
```

## Data

The parking-spot crops come from the dataset used in the [Computer Vision Engineer](https://www.youtube.com/watch?v=il8dMDlXrIE) image-classification tutorial. The download link is in the video description. The images are not included in this repo.

## Credits

Based on the image-classification tutorial by Computer Vision Engineer. This version splits training and inference into separate scripts, takes paths as arguments instead of a hard-coded Windows path, handles RGBA images, and adds a `predict.py` for classifying new crops.

## Stack

Python · scikit-learn · scikit-image · NumPy
