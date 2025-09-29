import idx2numpy

train_images_2d = idx2numpy.convert_from_file("dataset/train-images.idx3-ubyte") / 255.0
test_images_2d = idx2numpy.convert_from_file("dataset/t10k-images.idx3-ubyte") / 255.0

train_labels = idx2numpy.convert_from_file("dataset/train-labels.idx1-ubyte")
test_labels = idx2numpy.convert_from_file("dataset/t10k-labels.idx1-ubyte")
train_images = train_images_2d.reshape(train_images_2d.shape[0], -1)
test_images = test_images_2d.reshape(test_images_2d.shape[0], -1)
