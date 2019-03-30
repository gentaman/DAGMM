# %load test_dagm2007.py
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

x_train = np_imgs[:len(np_imgs)//2]
x_test = np_imgs[len(np_imgs)//2:len(np_imgs)//2+100]
np.random.seed(2)
x_test[:5] = np.random.rand(*x_test[:5].shape) + x_train.mean()

import gc
import pickle
import time

energies = []
times = []
for i in range(0, 51, 2):
    if i == 0:
        continue
    
    model_dagmm = DAGMM(
        comp_hiddens=[60,30,10, 1], comp_activation=tf.nn.tanh,
        est_hiddens=[10, 2], est_activation=tf.nn.tanh, est_dropout_ratio=0.25,
        epoch_size=3, minibatch_size=i
    )


    print('train size', len(x_train))
    e1 = time.time()
    model_dagmm.fit(x_train[np.random.permutation(len(x_train))])
    e2 = time.time()

    data = x_test
    energy = model_dagmm.predict(data)
    del model_dagmm
    gc.collect()
    energies.append(energy)
    times.append(e2 - e1)

with open('test_batch_size.log', mode='wb') as f:
    pickle.dump((energies, times), f)