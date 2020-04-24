import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from model import get_basic_model
from data import get_dataset
import edl

dataset = "MNIST"
(x_train, y_train), (x_test, y_test)  = get_dataset(dataset)

model = get_basic_model()
loss_function = edl.losses.Dirichlet_SOS
optimizer = tf.optimizers.Adam(learning_rate=5e-5)
epochs = 50

checkpoint_path ="models/"+ dataset + "/cp-{iteration:04d}.cpkt"

# training loop
batch_size = 64
for i in range(10000):
    idx = np.random.choice(x_train.shape[0], batch_size)
    x_input_batch = tf.gather(x_train, idx)

    with tf.GradientTape() as tape:
        outputs = model(x_input_batch) #forward pass
        alpha, probs = tf.split(outputs, 2, axis=-1)
        loss = loss_function(tf.gather(y_train, idx), alpha)

    grads = tape.gradient(loss, model.variables) #compute gradient
    optimizer.apply_gradients(zip(grads, model.variables))
    if i % 50 ==0:
        print("Training Loss: {}".format(loss.numpy().mean()))

    if i %1000 == 0:
        model.save_weights(checkpoint_path.format(iteration=i))
