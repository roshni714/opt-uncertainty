import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import csv

from metrics import expected_calibration_error, calibration_error, entropy, max_entropy
import opt_uncertainty
from opt_uncertainty import evidential as edl

# Load model
train_method = "softmax"

checkpoint_path = max(glob.glob('save/MNIST_{}/*.h5'.format(train_method)), key=os.path.getmtime) # get the last save model

model = tf.keras.models.load_model(checkpoint_path, custom_objects={
    'DenseDirichlet': edl.layers.DenseDirichlet,
    'DenseSoftmax': edl.layers.DenseSoftmax,
}, compile=False)


# Specify dataset, type of corruption, and corruption range
dataset = "MNIST"
#corruption_type = "rotation"
#corruption_range = [-360, 360]
corruption_type = "brightness"
corruption_range = [-0.65, 0.65]
#corruption_type= "blur"
#corruption_range = [0, 6]
corruption_amounts = np.linspace(corruption_range[0], corruption_range[1], 40)

# Get the mean uncertainty+accuracy  at a particular corruption level
stats = {"accuracy": [], "uncertainty": [], "corruption_level": list(corruption_amounts), "uncertainty_var":[], "expected_calibration_error": []}
for i in corruption_amounts:
    (_, _), (corr_x_test, y_test) = opt_uncertainty.data.get_dataset(dataset, corruption_type=corruption_type, corruption_level=i)
    outputs  = model(corr_x_test)

    if train_method == "evidence" or train_method == "evidence_regularized":
        alpha, probs = tf.split(outputs, 2, axis=-1)
    elif train_method == "softmax":
        logits, probs = tf.split(outputs, 2, axis=-1)
    uncertainty = entropy(probs)/max_entropy(10)
    u =  np.mean(uncertainty)
    u_var = np.var(uncertainty)

#    import pdb; pdb.set_trace()
    y_pred = tf.argmax(probs, axis=1)
    y_true = tf.argmax(y_test, axis=1)
    acc_preds = y_pred == y_true
    acc_preds = acc_preds.numpy()
    acc = np.mean(acc_preds)

    stats["accuracy"].append(acc)
    stats["uncertainty"].append(u)
    stats["uncertainty_var"].append(u_var)
    stats["expected_calibration_error"].append(expected_calibration_error(uncertainty, acc_preds))
    print("Corruption: {}, Acc: {}, Uncertainty: {}, Uncertainty Var: {}".format(i, acc, u, u_var))

print(stats)
f = open("results/{}_{}_{}.csv".format(dataset, train_method, corruption_type), 'w')
w = csv.DictWriter(f, stats.keys())
w.writeheader()
for i in range(len(corruption_amounts)):
    dic = {k: stats[k][i] for k in stats}
    w.writerow(dic)
f.close()

