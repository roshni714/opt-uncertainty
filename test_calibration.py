import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

import opt_uncertainty
from opt_uncertainty import evidential as edl

# Load model
checkpoint_path = max(glob.glob('save/MNIST/*.h5'), key=os.path.getmtime) # get the last save model

model = tf.keras.models.load_model(checkpoint_path, custom_objects={
    'DenseDirichlet':edl.layers.DenseDirichlet,
    'DenseSigmoid':edl.layers.DenseSigmoid,
})

# Specify dataset, type of corruption, and corruption range
dataset = "MNIST"
(x_train, y_train), (x_test, y_test) = opt_uncertainty.data.get_dataset(dataset)
corruption_type = "rotation"
corruption_range = [-20, 20]
#corruption_type = "brightness"
#corruption_range = [0.001, 2]

corruption_amounts = np.linspace(corruption_range[0], corruption_range[1], 20)

# Get the mean uncertainty+accuracy  at a particular corruption level
stats = {"accuracy": [], "uncertainty": [], "corruption_level": list(corruption_amounts)}
for i in corruption_amounts:
    corr_x_test = opt_uncertainty.data.apply_corruption_to_dataset(x_test, corruption_type, i)

    outputs  = model(corr_x_test)
    alpha, probs = tf.split(outputs, 2, axis=-1)
    u = np.mean(float(alpha.shape.as_list()[1]) / tf.reduce_sum(alpha, axis=1, keepdims=True))
    u_var = np.var(float(alpha.shape.as_list()[1]) / tf.reduce_sum(alpha, axis=1, keepdims=True))

    
    #uncertainty

    #TODO: should also record std dev of uncertainty
    y_pred = tf.argmax(probs, axis=1)
    y_true = tf.argmax(y_test, axis=1)

    acc = np.mean(y_pred == y_true)

    stats["accuracy"].append(acc)
    stats["uncertainty"].append(u)

    print("Corruption: {}, Acc: {}, Uncertainty: {}, Uncertainty Var: {}".format(i, acc, u, u_var))

print(stats)

#TODO plot corruption level vs. uncertainty, accuracy to visualize whether uncertainty is correlated with accuracy
