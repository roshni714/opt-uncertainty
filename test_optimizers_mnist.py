import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.interpolate

import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2

import opt_uncertainty


### Dataset
batch_size = 50
(xx, yy), (xx_test, yy_test) = tf.keras.datasets.mnist.load_data()
xx = np.expand_dims(xx, -1)/255. - 0.5
xx_test = np.expand_dims(xx_test, -1)/255. - 0.5
yy = tf.one_hot(yy, 10).numpy()
yy_test = tf.one_hot(yy_test, 10).numpy()

ds, ds_info = tfds.load('mnist_corrupted/motion_blur', split='test', with_info=True)
ds_numpy = list(tfds.as_numpy(ds))
xx_corr = np.array([ex["image"] for ex in ds_numpy])/255. - 0.5
yy_corr = tf.one_hot([ex["label"] for ex in ds_numpy], 10).numpy()

# for i in range(5):
#     plt.imshow(xx_corr[i,:,:,0])
#     plt.show()

track_inds = range(10)
for i in track_inds:
    cv2.imwrite("mnist_{}.png".format(i), (xx[i]+0.5)*255)

for i in track_inds:
    cv2.imwrite("mnistc_{}.png".format(i), (xx_corr[i]+0.5)*255)

### Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, 3, 2, padding="valid", activation="relu"),
    tf.keras.layers.Conv2D(64, 3, 2, padding="valid", activation="relu"),
    tf.keras.layers.Conv2D(64, 3, 2, padding="valid", activation="relu"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(10)
])


### Optimizer
optimizer = tf.keras.optimizers.SGD(5e-2)

### Forward pass
@tf.function
def forward(x):
    return model(x, training=True)

### Loss functions
@tf.function
def cross_entropy(y1, y2, reduce=True):
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y1, logits=y2)
    return loss


### Compute uncertainty
def compute_uncertainty(x, k=10):
    samples = [forward(x) for _ in range(k)]
    u = tf.math.reduce_std(samples, axis=0)
    u = tf.reduce_mean(u, axis=1)

    return u


### Training step
@tf.function
def train_step(x, y, unc_weighted=False, unc_all=None):
    with tf.GradientTape() as tape:
        y_ = forward(x)
        loss = cross_entropy(y, y_)
        if unc_weighted:
            u = compute_uncertainty(x)
            u_rank = tf.searchsorted(unc_all, u)
            u = tf.cast(u_rank, tf.float32) / unc_all.shape[0]
            loss *= u
        loss = tf.reduce_mean(loss)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


f, ax = plt.subplots(5,1)
ax = ax.flatten()
def redraw_plots():
    for a in ax: a.clear()

    N = 1000
    yy_ = forward(xx[:N])
    loss = cross_entropy(yy[:N], yy_)
    u_ = compute_uncertainty(xx[:N])

    u_test = compute_uncertainty(xx_test[:N])

    ax[0].plot(losses);
    ax[0].set_title("loss {:.3f}".format(np.mean(losses[-30:])))

    ax[1].plot(losses_test);
    ax[1].set_title("loss {:.3f}".format(np.mean(losses_test[-30:])))

    ax[2].plot(losses_corr);
    ax[2].set_title("loss {:.3f}".format(np.mean(losses_corr[-30:])))

    ax[3].hist(u_.numpy().flatten(), 30)
    ax[3].set_title("{:.4f} {:.4f}".format(u_.numpy().mean(), u_test.numpy().mean()))

    ax[4].plot(np.array(unc_per_images));

    f.canvas.draw()
    plt.pause(1e-18)



### Training loop
unc_weighted = False
# norm_func = lambda x: x
unc_all = None
losses = []
losses_test = []
losses_corr = []
unc_per_images = []
alpha = 0.9

for iter in tqdm(range(100000)):
    if iter % 100 == 0:
        # if unc_weighted:
        unc_all = compute_uncertainty(xx[:1000])
        unc_all = tf.sort(unc_all)


    i = np.random.choice(xx.shape[0], batch_size)
    loss = train_step(xx[i], yy[i], unc_weighted, unc_all)
    losses.append(loss.numpy())

    if iter % 20 == 0:
        losses_test.append(cross_entropy(yy_test[:200], forward(xx_test[:200])).numpy().mean())
        losses_corr.append(cross_entropy(yy_corr[:200], forward(xx_corr[:200])).numpy().mean())
        u = compute_uncertainty(xx[track_inds])
        u_rank = tf.searchsorted(unc_all, u)
        u = tf.cast(u_rank, tf.float32) / unc_all.shape[0]
        if len(unc_per_images) > 0:
            u_ = alpha*unc_per_images[-1] + (1-alpha)*u.numpy()
        else:
            u_ = u.numpy()
        unc_per_images.append(u_)

    if (iter+1) % 1000 == 0:
        redraw_plots()
        pass
    # if unc_weighted:
    #     print(outputs[0].numpy(), outputs[1].numpy().mean())
    # else:
    #     print(outputs.numpy())

# import pdb; pdb.set_trace()
dir = "mnist/ugd" if unc_weighted else "mnist/sgd"
np.savetxt("{}/loss_train.csv".format(dir), losses)
np.savetxt("{}/loss_test.csv".format(dir), losses_test)
np.savetxt("{}/loss_corr.csv".format(dir), losses_corr)
np.savetxt("{}/u_image.csv".format(dir), unc_per_images)

u = compute_uncertainty(xx[:5000])
u_test = compute_uncertainty(xx_test)
u_sort = tf.argsort(u_test).numpy()
for ii, i in enumerate(u_sort[:9]):
    cv2.imwrite("{}/low_{}.png".format(dir, ii), (xx_test[i]+0.5)*255)
for ii, i in enumerate(u_sort[-9:]):
    cv2.imwrite("{}/high_{}.png".format(dir, ii), (xx_test[i]+0.5)*255)

u_corr = compute_uncertainty(xx_corr)
np.savetxt("{}/u_train.csv".format(dir), u)
np.savetxt("{}/u_test.csv".format(dir), u_test)
np.savetxt("{}/u_corr.csv".format(dir), u_corr)





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
