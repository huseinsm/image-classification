from skimage.io import imread
from skimage.transform import resize

# Parking-spot crops are tiny, so 15x15 keeps enough signal while staying fast.
IMAGE_SIZE = (15, 15)
CATEGORIES = ["empty", "not_empty"]


def image_to_features(path):
    """Load an image and turn it into a flat feature vector for the SVM."""
    img = imread(path)
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    return resize(img, IMAGE_SIZE).flatten()
