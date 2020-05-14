import tensorflow as tf
import math
import scipy.ndimage as nd
import numpy as np
import sklearn.datasets as datasets

np.random.seed(0)

def randomly_invert(batch):
    new_batch = []
    for i in range(len(batch)):
        if i % 2 == 0:
            new_batch.append(-batch[i] + 1)
        else:
            new_batch.append(batch[i])
    return np.array(new_batch)

def get_toy_dataset(centers):

    X, y = datasets.make_moons(n_samples=6000, shuffle=True, noise=0.25, random_state=0)

    x_train, y_train = tf.cast(X[:5000], tf.float32), tf.one_hot(y[:5000], 2)
    x_test, y_test = tf.cast(X[5000:], tf.float32), tf.one_hot(y[5000:], 2)

    return (x_train, y_train), (x_test, y_test)


def get_dataset(dataset, corruption_type=None, corruption_level=None):
    """
    Generates train and test dataset with specified corruption applied to the
    test dataset.

    Parameters:
        dataset - name of dataset
    """
    if dataset.lower() == "mnist":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif dataset.lower() == "cifar10":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar.load_data()

    x_train  = (tf.cast(tf.expand_dims(x_train, -1), tf.float32)/255.)
    x_test  = (tf.cast(tf.expand_dims(x_test, -1), tf.float32)/255.)

        
    if corruption_type:
        x_test = apply_corruption_to_dataset(x_test, corruption_type, corruption_level)

    #normalize data
#    if dataset.lower() == "mnist":
#         if inversion_type == "all":
#             x_train = (x_train - 0.8693)/0.3081
#             x_test = (x_test - 0.8693)/0.3081
#             print(tf.reduce_mean(x_train))
#         else:
#             x_train = (x_train -0.1307)/0.3081
#             x_test = (x_test -0.1307)/0.3081

    y_train = tf.one_hot(y_train, 10)
    y_test = tf.one_hot(y_test, 10)

    return (x_train, y_train), (x_test, y_test)


def apply_corruption_to_dataset(imgs, corruption_type, corruption_level):

    if corruption_type == "rotation":
        angles = [corruption_level for i in range(imgs.shape[0])]
        corr_imgs = rotate_imgs(imgs, angles)
    elif corruption_type == "brightness":
        corr_imgs = tf.image.adjust_brightness(imgs, corruption_level)
        corr_imgs = tf.clip_by_value(corr_imgs, 0, 1)
    elif corruption_type == "blur":
        blurs = [corruption_level for i in range(imgs.shape[0])]
        corr_imgs = blur_imgs(imgs, blurs)
    return corr_imgs


def blur_imgs(imgs, blurs):
    test_imgs = []
    for i in range(len(imgs)):
        s = imgs[i].shape
        nimg = nd.gaussian_filter(imgs[i].numpy().reshape(s[0], s[1]), blurs[i]).reshape(s[0], s[1], s[2])
        nimg = np.clip(a=nimg, a_min=0, a_max=1)
        test_imgs.append(nimg)
    return np.array(test_imgs)

def rotate_imgs(imgs, angles):
    print("Rotating imgs...")
    test_imgs = []
    for i in range(len(imgs)):
        s = imgs[i].shape
        nimg = nd.rotate(imgs[i].numpy().reshape(s[0], s[1]), angles[i], reshape=False).reshape(s[0], s[1], s[2])
        nimg = np.clip(a=nimg, a_min=0, a_max=1)
        test_imgs.append(nimg)
    return np.array(test_imgs)
