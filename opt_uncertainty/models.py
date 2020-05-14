import tensorflow as tf

from . import evidential as edl


def get_basic_model(input_shape, method="evidence"):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, 5, 2, 'valid', activation='relu', input_shape=input_shape),
        tf.keras.layers.Conv2D(64, 3, 2, 'valid', activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
    ])
    if method == "softmax":
        print("softmax model")
        model.add(edl.layers.DenseSoftmax(10))
    elif method == "evidence" or method == "evidence_regularized":
        print("evidence model")
        model.add(edl.layers.DenseDirichlet(10))
    else:
        raise ValueError("unrecognized output method: {}".format(method))

    return model

def get_toy_model(input_shape, num_classes, method="evidence"):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(100, activation='relu'),
    ])
    if method == "softmax":
        print("softmax model")
        model.add(edl.layers.DenseSoftmax(num_classes))
    elif method == "evidence" or method == "evidence_regularized":
        print("evidence model")
        model.add(edl.layers.DenseDirichlet(num_classes))
    else:
        raise ValueError("unrecognized output method: {}".format(method))

    return model
