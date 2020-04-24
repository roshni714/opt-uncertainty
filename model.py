import tensorflow as tf
import edl


def get_basic_model():
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, 5, 2, 'valid', activation='relu'),
    tf.keras.layers.Conv2D(64, 3, 2, 'valid', activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    edl.layers.DenseDirichlet(10)
    ])
    return model


