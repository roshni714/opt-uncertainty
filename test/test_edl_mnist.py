import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from opt_uncertainty import evidential as edl


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = tf.cast(tf.expand_dims(x_train, -1), tf.float32)/255.
x_test = tf.cast(tf.expand_dims(x_test, -1), tf.float32)/255.
y_train = tf.one_hot(y_train, 10)
y_test = tf.one_hot(y_test, 10)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, 5, 2, 'valid', activation='relu'),
    tf.keras.layers.Conv2D(64, 3, 2, 'valid', activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    edl.layers.DenseDirichlet(10)
])

# loss_function = edl.losses.Softmax_CE
loss_function = edl.losses.Dirichlet_SOS
optimizer = tf.keras.optimizers.Adam(1e-4)

f, ax = plt.subplots(3,1)
ax = ax.flatten()

test_img = x_test[np.where(y_test.numpy()[:,1])[0][0]].numpy()
def rotate_img(x, deg):
    import scipy.ndimage as nd
    return nd.rotate(x.reshape(28,28),deg,reshape=False).ravel()

test_imgs = []
N_imgs = 20
for i,deg in enumerate(np.linspace(0,360,N_imgs)):
    nimg = rotate_img(test_img,deg).reshape(28,28,1)
    nimg = np.clip(a=nimg,a_min=0,a_max=1)
    test_imgs.append(nimg)
test_imgs = np.array(test_imgs)

def redraw_plots():
    for a in ax: a.clear()

    outputs = model(test_imgs)
    alpha, prob = tf.split(outputs, 2, axis=-1)
    u = float(alpha.shape.as_list()[1]) / tf.reduce_sum(alpha, axis=1, keepdims=True) #uncertainty

    ax[0].plot(u.numpy()); ax[0].set_xlim(0,N_imgs-1); ax[0].set_ylim(-0.1,1.1); ax[0].set_title("Uncertainty")
    ax[1].plot(prob[:,1]); ax[1].set_xlim(0,N_imgs-1); ax[1].set_ylim(-0.1,1.1);  ax[1].set_title("P('1')")
    ax[2].imshow(np.hstack(test_imgs)[:,:,0], cmap='gray', vmin=0, vmax=1); ax[2].set_title("Input")

    f.canvas.draw()
    plt.pause(1e-9)


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
        redraw_plots()
        print(loss.numpy().mean())

import pdb; pdb.set_trace()
