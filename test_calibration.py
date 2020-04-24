import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from data import get_dataset, apply_corruption_to_dataset
from model import get_basic_model
import edl

#Load model
checkpoint path = "models/MNIST/cp-10000.ckpt" 
model = get_basic_model()
model.load(checkpoint_path)

# Specify dataset, type of corruption, and corruption range
dataset = "MNIST"
(x_train, y_train), (x_test, y_test) = get_dataset(dataset)
corruption_type = "rotation"
corruption_range = [-360, 360]
#corruption_type = "brightness"
#corruption_range = [0.001, 2]

corruption_amounts = np.linspace(corruption_range[0], corruption_range[1], 20)

# Get the mean uncertainty+accuracy  at a particular corruption level
stats = {"accuracy": [], "uncertainty": [], "corruption_level": list(corruption_amounts)}
for i in corruption_amounts:
    corr_x_test = apply_corruption_to_dataset(x_test, corruption_type, i) 

    outputs  = model(corr_x_test)
    alpha, probs = tf.split(outputs, 2, axis=-1)
    u = float(alpha.shape.as_list()[1]) / tf.reduce_sum(alpha, axis=1, keepdims=True) #uncertainty

    #TODO: should also record std dev of uncertainty
    y_pred = tf.argmax(probs, axis=1)
    y_true = tf.argmax(y_test, axis=1)

    acc = tf.reduce_mean(y_pred == y_true)

    stats["accuracy"].append(acc)
    stats["uncertainty"].append(u)


#TODO plot corruption level vs. uncertainty, accuracy to visualize whether uncertainty is correlated with accuracy 

