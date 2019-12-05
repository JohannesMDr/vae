from keras.models import Model
from keras.layers import *
from keras.layers.convolutional import Convolution3D, MaxPooling3D, AveragePooling3D
from keras.layers.normalization import BatchNormalization
from keras.metrics import binary_crossentropy
from keras import backend as K
import numpy as np


def build_model(input_shape, latent_dim, optimizer=Adam(lr=8e-4, beta_1=0.5, beta_2=0.9)):
  #encoder
  input_img = Input(shape=input_shape)

  # prepare for resblock
  X0 = input_img
  X0 = Conv3D(n_filter, (3, 3, 3),  padding="same", kernel_initializer='he_normal')(X0)

  # resblock
  #like resnet101
  X1 = ResBlock(X0, 64,  3,  1)
  X2 = ResBlock(X1, 128, 4,  2)
  X3 = ResBlock(X2, 256, 23, 3)
  x  = ResBlock(X3, 512, 3,  4)

  x = BatchNormalization()(x)

  # VAE
  shape_before_flattening = K.int_shape(x)

  x = Flatten()(x)

  z_mean = Dense(latent_dim)(x)
  z_log_var = Dense(latent_dim)(x)

  def sampling(args):
      z_mean, z_log_var = args
      epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),mean=0., stddev=1.)
      return z_mean + K.exp(z_log_var) * epsilon

  z = Lambda(sampling)([z_mean, z_log_var])

  encoder = Model(input_img, z)
  # encoder.summary()
  
  
  # decoder
  # This is the input where we will feed `z`.
  decoder_input = layers.Input(K.int_shape(z)[1:])

  # Upsample to the correct number of units
  X_0 = Dense(np.prod(shape_before_flattening[1:]))(decoder_input)

  # Reshape into an image of the same shape as before our last `Flatten` layer
  X_0 = Reshape(shape_before_flattening[1:])(X_0)

  # prepare for resblock + upsampling
  X_0 = ReLU()(X_0)

  # resblock + upsampling
  X_1 = ResBlock(X_0, 512, 3, 0)
  X_1 = UpSampling3D((2, 2, 2))(X_0)
  X_1 = Conv3D(256, (3, 3, 3), padding='same', kernel_initializer='he_normal')(X_1)
  X_2 = ResBlock(X_1, 256, 23, 0)
  X_2 = UpSampling3D((2, 2, 2))(X_2)
  X_2 = Conv3D(128, (3, 3, 3), padding='same', kernel_initializer='he_normal')(X_2)
  X_3 = ResBlock(X_2, 128, 4, 0)
  X_3 = UpSampling3D((2, 2, 2))(X_3)
  X_3 = Conv3D(64, (3, 3, 3), padding='same', kernel_initializer='he_normal')(X_3)
  X_4 = ResBlock(X_3, 64, 3, 0)
  x   = UpSampling3D((2, 2, 2))(X_3)

  x = Conv3D(1, kernel_size=(3, 3, 3), activation='sigmoid', padding='same',  kernel_initializer='he_normal', name='decoder_output')(x)

  # We end up with a feature map of the same size as the original input.

  # This is our decoder model.
  decoder = Model(decoder_input, x)
  # decoder.summary()

  # We then apply it to `z` to recover the decoded `z`.
  z_decoded = decoder(z)
  
  def vae_loss(x, z_decoded):
    x = K.flatten(x)
    z_decoded = K.flatten(z_decoded)
    xent_loss = binary_crossentropy(x, z_decoded)
    kl_loss = -5e-4 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=1)
    return K.mean(xent_loss + kl_loss)
  
  # encoder + decoder + loss_function
  vae = Model(input_img, z_decoded)
  vae.compile(optimizer=optimizer, loss=vae_loss)
  # vae.summary()
  
  return encoder, decoder, vae
