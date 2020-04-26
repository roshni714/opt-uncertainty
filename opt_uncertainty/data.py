import tensorflow as tf
import math
import scipy.ndimage as nd
import numpy as np

def get_dataset(dataset):
    """
    Generates train and test dataset with specified corruption applied to the
    test dataset.

    Parameters:
        dataset - name of dataset
        dictionary - dictionary specifying amount of corruption
    """
    if dataset.lower() == "mnist":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif dataset.lower() == "cifar10":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar.load_data()

    x_train  = tf.cast(tf.expand_dims(x_train, -1), tf.float32)/255.
    x_test  = tf.cast(tf.expand_dims(x_test, -1), tf.float32)/255.
    y_train = tf.one_hot(y_train, 10)
    y_test = tf.one_hot(y_test, 10)

    return (x_train, y_train), (x_test, y_test)


def apply_corruption_to_dataset(imgs, corruption_type, corruption_level):

    if corruption_type == "rotation":
        angles = [corruption_level * math.pi/180. for i in range(imgs.shape[0])]
        corr_imgs = rotate_imgs(imgs, angles)
    elif corruption_type == "brightness":
        corr_imgs = tf.image.adjust_brightness(imgs, corruption_level)

    return corr_imgs


def rotate_imgs(imgs, angles):
    print("Rotating imgs...")
    test_imgs = []
    for i in range(len(imgs)):
        s = imgs[i].shape
        nimg = nd.rotate(imgs[i].numpy().reshape(s[0], s[1]), angles[i], reshape=False).reshape(s[0], s[1], s[2])
        nimg = np.clip(a=nimg, a_min=0, a_max=1)
        test_imgs.append(nimg)
    return np.array(test_imgs)


