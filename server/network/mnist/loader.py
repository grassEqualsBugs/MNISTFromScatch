import idx2numpy
import numpy as np
from numpy.typing import NDArray


def load_mnistdata():
    train_images_2d = (
        idx2numpy.convert_from_file("network/mnist/dataset/train-images.idx3-ubyte")
        / 255.0
    )
    test_images_2d = (
        idx2numpy.convert_from_file("network/mnist/dataset/t10k-images.idx3-ubyte")
        / 255.0
    )

    return {
        "train_labels": idx2numpy.convert_from_file(
            "network/mnist/dataset/train-labels.idx1-ubyte"
        ),
        "test_labels": idx2numpy.convert_from_file(
            "network/mnist/dataset/t10k-labels.idx1-ubyte"
        ),
        "train_images": train_images_2d.reshape(train_images_2d.shape[0], -1),
        "test_images": test_images_2d.reshape(test_images_2d.shape[0], -1),
    }


def ascii_image(image: NDArray[np.float64]):
    image = image.reshape(28, 28)
    for row in image:
        for pixel in row:
            if pixel > 0.5:
                print("#", end="")
            else:
                print(" ", end="")
        print()
