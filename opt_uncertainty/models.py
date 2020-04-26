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
        model.add(edl.layers.DenseSoftmax(10))
    elif method == "evidence":
        model.add(edl.layers.DenseDirichlet(10))
    else:
        raise ValueError("unrecognized output method: {}".format(method))

    return model
