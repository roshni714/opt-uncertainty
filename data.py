import tensorflow.tf

def get_dataset(dataset):
    """
    Generates train and test dataset with specified corruption applied to the
    test dataset.

    Parameters:
        dataset - name of dataset
        dictionary - dictionary specifying amount of corruption
    """
    if dataset == "MNIST":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif dataset== "CIFAR10":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar.load_data()

    x_train, x_test = x_train/255., x_test/255.
    y_train = tf.one_hot(y_train, 10)
    y_test = tf.one_hot(y_test, 10)

    return (x_train, y_train), (x_test, y_test)


def apply_corruption_to_dataset(imgs, corruption_type, corruption_level):

    if corruption_type == "rotation":
        corr_imgs = tf.contrib.image.rotate(imgs,  corruption_level * math.pi/180., interpolation='BILINEAR')
    elif corruption_type == "brightness":
        corr_imgs = tf.image.adjust_brightness(imgs, corruption_level)

    return corr_imgs
