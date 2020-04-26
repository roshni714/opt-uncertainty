import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense
from tensorflow.keras.models import Sequential



class DenseSoftmax(Layer):
    def __init__(self, units):
        super(DenseSoftmax, self).__init__()
        self.units = int(units)
        self.dense = Dense(int(units))

    def call(self, x):
        logits = self.dense(x)
        prob = tf.nn.softmax(logits)
        return tf.concat([logits, prob], axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 2*self.units)

    def get_config(self):
        return {'units': self.units}



class DenseDirichlet(Layer):
    def __init__(self, units):
        super(DenseDirichlet, self).__init__()
        self.units = int(units)
        self.dense = Dense(int(units))

    def call(self, x):
        output = self.dense(x)
        evidence = tf.exp(output)
        alpha = evidence + 1
        prob = alpha / tf.reduce_sum(alpha, 1, keepdims=True)
        return tf.concat([alpha, prob], axis=-1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 2*self.units)

    def get_config(self):
        return {'units': self.units}
