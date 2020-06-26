import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
from metrics import entropy
import opt_uncertainty
from opt_uncertainty import evidential as edl

LOSS_FUNCTIONS = {"softmax": edl.losses.Softmax_CE,
                  "evidence": edl.losses.Dirichlet_SOS,
                  "evidence_regularized": edl.losses.Dirichlet_Regularized_SOS}

dataset = "toy"
train_method = "evidence" #"evidence" #"softmax"
(x_train, y_train), (x_test, y_test) = opt_uncertainty.data.get_toy_dataset(dataset)

model = opt_uncertainty.models.get_toy_model(input_shape=x_train.shape[1:], num_classes=2, method=train_method)
loss_function = LOSS_FUNCTIONS[train_method]
optimizer = tf.optimizers.Adam(1e-4)

epsilon = 1e-2
v = 1

@tf.function # move to tf graph for speed
def train_step(x, y, v=1, epsilon=epsilon):
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
        v += 1e-3 * (loss_reg - epsilon)

    return loss, v


NAME = {"evidence_regularized": "Evidence Regularized",
        "evidence": "Evidence",
        "softmax": "Softmax"}

def make_uncertainty_plot(uncertainty, iteration):
    fig = plt.figure(figsize=(11.25, 5))
    ax = fig.subplots(1, 3, gridspec_kw={"width_ratios": [5, 5, 0.25]})

    ax[0].scatter(x_test[:, 0], x_test[:, 1], c=np.argmax(y_test, axis=1))
    ax[0].set_title("Test Distribution")

    res = ax[1].scatter(x_test[:, 0], x_test[:, 1], c=uncertainty, vmin=0, vmax=0.75)
    ax[1].set_title("{}".format(NAME[train_method]))
    fig.colorbar(res, cax=ax[2])
    ax[2].set_title("Uncertainty")

    plt.savefig("figs/toy/{}_{}.pdf".format(train_method, iteration))
    plt.close()

def make_training_plot(vs, losses, accs, train_method, iteration):
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

    plt.savefig("figs/training/{}_{}.pdf".format(train_method,i))
    plt.close()

# training loop
batch_size = 32
num_iters = 50000
v=1

vs = []
losses = []
accs = []

found_nan = False
for i in range(num_iters):
    loss, v = train_step(x_train, y_train, v)

    if i % 200 == 0:
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
       make_training_plot(vs, losses, accs, train_method, i)
       print("Test Accuracy [{}/{}]: {}".format(i, num_iters, acc))

    if i % 5000 == 0:
       outputs  = model(x_test)
       _,  probs = tf.split(outputs, 2, axis=-1)
       y_pred = tf.argmax(probs, axis=1)
       y_true = tf.argmax(y_test, axis=1)
       acc = np.mean(y_pred==y_true)

       uncertainty = entropy(probs)
       make_uncertainty_plot(uncertainty, i)

