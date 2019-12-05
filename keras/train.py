from keras.models import Model
from keras.layers import *
from keras.layers.normalization import BatchNormalization
from keras.metrics import binary_crossentropy
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import pickle

from .model import build_model
from .generators import MyGenerator


# path
path = 'drive/My Drive'
path_data = 'drive/My Drive/patient_data'
path_dataset = 'drive/My Drive/VAE/dataset'
model_name = 'vae4-1'
path_model = f'drive/My Drive/VAE/model/{model_name}'
os.makedirs(path_model, exist_ok=True)

# load data
x_input = np.load(path_dataset + '/dataset2.npy').reshape(23*500, 16, 64, 64, 1)
x_train = x_input[:11000]
x_val = x_input[11000:]


# hyperparameter
x_size, y_size, z_size = (64, 64, 16)
latent_dim = 100  # Dimensionality of the latent space: a plane

# network parameters
input_shape = (z_size, x_size, y_size, 1)
n_filter = 64
batch_size = 4
epochs = 40


# build model
encoder, decoder, vae = build_model(input_shape=input_shape, latent_dim=latent_dim)
train_gen = MyGenerator(x_train, batch_size)
val_gen = MyGenerator(x_val, batch_size)


# training
# training
history = vae.fit_generator(
        train_gen, 
        steps_per_epoch=train_gen.num_batches_per_epoch, 
        validation_data=val_gen, 
        validation_steps=val_gen.num_batches_per_epoch,
        shuffle=True,
        epochs=epochs,
        callbacks=[model_checkpoint])
vae.save(os.path.join(path_model, '{}_ep40.h5'.format(model_name)))
pickle.dump(history, open(os.path.join(path_model, 'history.pkl'), 'wb'))
