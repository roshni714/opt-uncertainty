import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.interpolate

import opt_uncertainty


unc_weighted = False


### Dataset
batch_size = 50
def f(x):
    return np.sin(3*x)/(3*x)
xx = np.concatenate([np.linspace(-3,3,200), np.linspace(-1,1,600)]).reshape(-1,1).astype(np.float32)
xx = xx[np.random.choice(xx.shape[0], xx.shape[0], replace=False)]
yy = f(xx)

xx_test = np.linspace(-3,3,1000).reshape(-1,1).astype(np.float32)
xx_test = xx_test[np.random.choice(xx_test.shape[0], xx_test.shape[0], replace=False)]
yy_test = f(xx_test)


plt.scatter(xx[::5], yy[::5])
plt.savefig("toy_train.pdf")
plt.clf()

plt.scatter(xx_test[::5], yy_test[::5])
plt.savefig("toy_test.pdf")
plt.clf()
# yy += np.random.randn(*xx.shape)/50.


### Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(1)
])

### Optimizer
optimizer = tf.keras.optimizers.SGD(2e-2)
# optimizer = tf.keras.optimizers.Adam(5e-4)

### Forward pass
@tf.function
def forward(x, dropout):
    return model(x, training=dropout)

### Mean squared error
@tf.function
def mse(y1, y2):
    return tf.reduce_mean(tf.square(y1 - y2), axis=1, keepdims=True)

### Compute uncertainty
def compute_uncertainty(x, k=10):
    samples = [forward(x, True) for _ in range(k)]
    u = tf.math.reduce_std(samples, axis=0)
    u = tf.reduce_mean(u, axis=1)

    return u


### Training step
@tf.function
def train_step(x, y, unc_weighted=False, unc_all=None):
    with tf.GradientTape() as tape:
        y_ = forward(x, unc_weighted)
        loss = mse(y, y_)
        if unc_weighted:
            u = compute_uncertainty(x)
            u_rank = tf.searchsorted(unc_all, u)
            u = tf.cast(u_rank, tf.float32) / u.shape[0]
            loss *= u
        loss = tf.reduce_mean(loss)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


fig, ax = plt.subplots(4,1)
ax = ax.flatten()
def redraw_plots():
    for a in ax: a.clear()

    yy_ = model(xx)
    loss = (yy_-yy)**2
    u_ = compute_uncertainty(xx)

    ax[0].scatter(xx, yy, s=1, c="r");
    ax[0].scatter(xx, yy_, s=6, c=loss)
    ax[0].set_title("loss")

    ax[1].scatter(xx, yy, s=1, c="r");
    ax[1].scatter(xx, yy_, s=6, c=u_[:,np.newaxis])
    ax[1].set_title("unc")

    ax[2].hist(u_.numpy().flatten(), 30)
    ax[2].set_title(u_.numpy().mean())

    ax[3].plot(np.array(losses)[:,0])
    ax[3].plot(np.array(losses_test)[:,0])
    ax[3].set_title("{:.4f} {:.4f}".format(np.mean(losses[-50:]), np.mean(losses_test[-50:])))
    ax[3].set_yscale('log')

    fig.canvas.draw()
    plt.pause(1e-18)



### Training loop
# norm_func = lambda x: x
unc_all = None
losses = []
losses_test = []
for iter in tqdm(range(100000)):
    if iter % 100 == 0:
        if unc_weighted:
            unc_all = compute_uncertainty(xx)
            unc_all = tf.sort(unc_all)


    i = np.random.choice(xx.shape[0], batch_size)
    loss = train_step(xx[i], yy[i], unc_weighted, unc_all)

    if iter % 20 == 0:
        N = 400
        yy_ = forward(xx[:N], False)
        u_ = compute_uncertainty(xx[:N])
        losses.append([mse(yy[:N], yy_).numpy().mean(), u_.numpy().mean()])

        yy_test_ = forward(xx_test[:N], False)
        u_test = compute_uncertainty(xx_test[:N])
        losses_test.append([mse(yy_test[:N], yy_test_).numpy().mean(), u_test.numpy().mean()])

    if (iter+1) % 2000 == 0:
        redraw_plots()
        pass
    # if unc_weighted:
    #     print(outputs[0].numpy(), outputs[1].numpy().mean())
    # else:
    #     print(outputs.numpy())

np.savetxt("toy_loss_train_{}.csv".format(unc_weighted), losses)
np.savetxt("toy_loss_test_{}.csv".format(unc_weighted), losses_test)

xx_test = np.linspace(-3,3,1000).reshape(-1,1).astype(np.float32)
yy_test = f(xx_test)
yy_test_ = forward(xx_test, False)
u_test = compute_uncertainty(xx_test)
np.savetxt("toy_true_{}.csv".format(unc_weighted), yy_test)
np.savetxt("toy_pred_{}.csv".format(unc_weighted), yy_test_.numpy())
np.savetxt("toy_u_{}.csv".format(unc_weighted), u_test.numpy())

# yy_ = model(xx)
# loss = (yy_-yy)**2
# u_ = compute_uncertainty(xx, norm=False)
# print("unc", u_.numpy().mean())
# plt.scatter(xx, yy, s=1, c="r")
# plt.scatter(xx, yy_, c=loss)
# plt.show()
#
# plt.hist(u_.numpy().flatten(), 30)
# plt.show()
