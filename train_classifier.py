import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

import opt_uncertainty
from opt_uncertainty import evidential as edl

dataset = "MNIST"
(x_train, y_train), (x_test, y_test) = opt_uncertainty.data.get_dataset(dataset)

model = opt_uncertainty.models.get_basic_model(input_shape=x_train.shape[1:], method="vanilla")
loss_function = edl.losses.Dirichlet_SOS
optimizer = tf.optimizers.Adam(1e-4)
epochs = 50

checkpoint_path = os.path.join("save", dataset, "cp-{iteration:04d}.h5")

@tf.function # move to tf graph for speed
def train_step(x, y):
    with tf.GradientTape() as tape:
        outputs = model(x) #forward pass
        alpha, probs = tf.split(outputs, 2, axis=-1)
        loss = loss_function(y, alpha)

    grads = tape.gradient(loss, model.variables) #compute gradient
    optimizer.apply_gradients(zip(grads, model.variables))
    return loss


# training loop
batch_size = 64
num_iters = 100000
for i in range(num_iters):
    idx = np.random.choice(x_train.shape[0], batch_size)
    x_input_batch = tf.gather(x_train, idx)
    y_input_batch = tf.gather(y_train, idx)
    loss = train_step(x_input_batch, y_input_batch)

    if i % 50 == 0:
        print("[{}/{}]: {}".format(i, num_iters, loss.numpy().mean()))

    if i % 1000 == 0:
       outputs  = model(x_test)
       alpha, probs = tf.split(outputs, 2, axis=-1)
       y_pred = tf.argmax(probs, axis=1)
       y_true = tf.argmax(y_test, axis=1)
       acc = np.mean(y_pred==y_true)
       print("Test Accuracy [{}/30000]: {}".format(i, acc))

    if i % 10000 == 0:
       model.save(checkpoint_path.format(iteration=i))
