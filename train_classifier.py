import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

import opt_uncertainty
from opt_uncertainty import evidential as edl


LOSS_FUNCTIONS = {"softmax": edl.losses.Softmax_CE,
                  "evidence": edl.losses.Dirichlet_SOS,
                  "evidence_regularized": edl.losses.Dirichlet_Regularized_SOS}

dataset = "MNIST"
train_method = "evidence_regularized" #"evidence"
(x_train, y_train), (x_test, y_test) = opt_uncertainty.data.get_dataset(dataset)


model = opt_uncertainty.models.get_basic_model(input_shape=x_train.shape[1:], method=train_method)
loss_function = LOSS_FUNCTIONS[train_method]
optimizer = tf.optimizers.Adam(1e-4)
epochs = 50

checkpoint_path = os.path.join("save", "{}_{}".format(dataset, train_method), "cp-{iteration:04d}.h5")

@tf.function # move to tf graph for speed
def train_step(x, y, v=1, epsilon=0.002):
    with tf.GradientTape() as tape:
        outputs = model(x) #forward pass
        alpha, probs = tf.split(outputs, 2, axis=-1)
        if train_method == "evidence_regularized":
            loss, loss_reg = loss_function(y, alpha, v, epsilon)
        else:
            loss  = loss_function(y, alpha)

    grads = tape.gradient(loss, model.variables) #compute gradient
    optimizer.apply_gradients(zip(grads, model.variables))

    if train_method =="evidence_regularized":
        v = tf.maximum(v + 5e-3 * (loss_reg - epsilon), 0.001)

    return loss, v

def make_training_plot(vs, losses, accs, iteration):
    fig = plt.figure(figsize=(15,5))
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    x_vs, y_vs = zip(*vs)
    x_losses, y_losses = zip(*losses)
    x_accs, y_accs = zip(*accs)

    ax1.plot(x_vs, y_vs, label="Regularization Parameter")
    ax2.plot(x_losses, y_losses, label="Training Loss")
    ax3.plot(x_accs, y_accs, label="Test Accuracy")

    ax1.set_title("Regularization Parameter")
    ax2.set_title("Training Loss")
    ax3.set_title("Test Accuracy")

    ax1.set_xlabel("Iteration")
    ax2.set_xlabel("Iteration")
    ax3.set_xlabel("Iteration")

    plt.savefig("figs/gda/training_{}.pdf".format(i))


# training loop
batch_size = 64
num_iters = 50000
v=1

vs = []
losses = []
accs = []

for i in range(num_iters):
    idx = np.random.choice(x_train.shape[0], batch_size)
    x_input_batch = tf.gather(x_train, idx)
    y_input_batch = tf.gather(y_train, idx)
    loss, v = train_step(x_input_batch, y_input_batch, v)

   
    if i % 50 == 0:
        print("[{}/{}] Loss: {}".format(i, num_iters, loss.numpy().mean()))
        print("[{}/{}] v: {}".format(i, num_iters, v.numpy().mean()))
        vs.append((i, v.numpy().mean()))
        losses.append((i, loss.numpy().mean()))

    if i % 1000 == 0:
       outputs  = model(x_test)
       alpha, probs = tf.split(outputs, 2, axis=-1)
       y_pred = tf.argmax(probs, axis=1)
       y_true = tf.argmax(y_test, axis=1)
       acc = np.mean(y_pred==y_true)
       accs.append((i, acc))
       print("Test Accuracy [{}/{}]: {}".format(i, num_iters, acc))

    if i % 10000 == 0:
       model.save(checkpoint_path.format(iteration=i))
       make_training_plot(vs, losses, accs, iteration=i)


