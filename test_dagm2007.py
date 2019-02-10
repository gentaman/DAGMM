import os
from PIL import Image

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from dagmm import DAGMM

import matplotlib.pyplot as plt

root = './data/Class1/'
fnames = os.listdir(root)
np_imgs = []

for fname in fnames:
    im = Image.open(os.path.join(root, fname))
    np_imgs.append(np.asarray(im))
np_imgs = np.asarray(np_imgs)
np_imgs = np_imgs.astype(np.float64)

np_imgs_mean = np_imgs.mean(axis=0)
np_imgs_var = np_imgs.var(axis=0)
np_imgs = (np_imgs - np_imgs_mean) / np.sqrt(np_imgs_var)
np_imgs = np_imgs.astype(np.float64)

np_imgs = np_imgs.reshape(-1, 512*512)


model_dagmm = DAGMM(
    comp_hiddens=[16,8,1], comp_activation=tf.nn.tanh,
    est_hiddens=[8,4], est_activation=tf.nn.tanh, est_dropout_ratio=0.25,
    epoch_size=100, minibatch_size=100
)

x_train = np_imgs[:len(np_imgs)//2]
x_test = np_imgs[len(np_imgs)//2:len(np_imgs)//2+100]
np.random.seed(0)
x_test[:5] = np.random.rand(*x_test[:5].shape) + x_train.mean()

print('train size', len(x_train))
model_dagmm.fit(x_train)

data = x_test
energy = model_dagmm.predict(data)

print(energy.shape)

plt.figure(figsize=[16,6])
histinfo = plt.hist(energy, bins=50)
plt.xlabel("DAGMM Energy")
plt.ylabel("Number of Sample(s)")
plt.savefig("./dagm2007_class1_energy_hist.png")
plt.show()

plt.figure(figsize=[16,6])
plt.plot(np.concatenate([np.ones(5)*energy[5:].min(), energy[5:]]), "o-")
plt.plot(energy[:5], "o-", c='g')
plt.hlines(np.percentile(energy, 95), 0, 100, 'g', linestyles='dashed')
plt.xlabel("Index (row) of Sample")
plt.ylabel("Energy")
plt.savefig("./dagm2007_class1_energy.png")
plt.show()

fig, axes = plt.subplots(nrows=5, ncols=5, figsize=[16,16], sharex=True, sharey=True)
plt.subplots_adjust(wspace=0.05, hspace=0.05)

for row in range(5):
    for col in range(5):
        ax = axes[row, col]
        if row != col:
            ax.plot(data[5:,col], data[5:,row], ".", c='b')
            ax.plot(data[:5,col], data[:5,row], ".", c='g')
            ano_index = np.arange(len(energy))[energy > np.percentile(energy, 95)]
            ax.plot(data[ano_index,col], data[ano_index,row], "x", c="r", alpha=0.5, markersize=8)
plt.tight_layout()
plt.savefig("./dagm2007_scatter.png")
plt.show()
