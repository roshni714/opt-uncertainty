import tensorflow as tf
from tensorflow.python.keras.optimizer_v2 import optimizer_v2


class SGD(optimizer_v2.OptimizerV2):
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def apply_gradients(self, grads_and_vars):
        for g, v in grads_and_vars:
            new_v = v - self.learning_rate * g
            v.assign(new_v)

    def get_config(self):
        config = super(SGD, self).get_config()
        config.update({
            "learning_rate": self.learning_rate,
        })
        return config



class UASGD(optimizer_v2.OptimizerV2):
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def apply_gradients(self, grads_and_vars):
        for g, v in grads_and_vars:
            new_v = v - self.learning_rate * g
            v.assign(new_v)

    def get_config(self):
        config = super(SGD, self).get_config()
        config.update({
            "learning_rate": self.learning_rate,
        })
        return config
