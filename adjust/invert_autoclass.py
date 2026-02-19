# -*- coding: utf-8 -*-
"""
Created on Mon Dec 3 15:35:29 2020
Modified on Fri Jun 13 2025

@author: hli45 (Original), Modified by Preston Raab
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
import time
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# --- Gradient Reversal Layer ---
@tf.custom_gradient
def grad_reverse(x):
    """
    Implements a gradient reversal layer. No effect on the forward pass.
    """
    return (tf.identity(x), lambda dy: -dy)

class GradientReversal(tf.keras.layers.Layer):
    """
    Wrapper for the gradient reversal function to be used as a Keras layer.
    """
    def __init__(self):
        super().__init__()

    def call(self, x):
        return grad_reverse(x)


def take_norm(data, cellwise_norm=True, log1p=True):
    data_norm = data.copy()
    data_norm = data_norm.astype('float32')
    if cellwise_norm:
        libs = data.sum(axis=1)
        norm_factor = np.diag(np.median(libs) / libs)
        data_norm = np.dot(norm_factor, data_norm)

    if log1p:
        data_norm = np.log2(data_norm + 1.)
    return data_norm


def BatchCorrectImpute(data, batch_info, cellwise_norm=True, log1p=True,
                       encoder_layer_size=[128], dropout_rate=0.1, adversarial_weight=0.6,
                       reg=0.000, batch_size=32, epochs=300,
                       verbose=False, es=30, lr=15):
    """Adversarial Autoencoder for batch effect correction and imputation.
    This network learns a batch-invariant embedding to impute gene expression
    data while removing batch effects. An adversary network branch is trained
    to predict batch origin, while the encoder is trained to fool the adversary.

    Parameters
    ----------
    data: numpy matrix of raw counts.
          Each row represents a cell, each column represents a gene.
    batch_info: 'tuple' or 'list'.
          A list containing the batch ID for each cell. This is required.
    cellwise_norm: 'bool'. Default: True.
          If True, library size normalization is performed.
    log1p: 'bool'. Default: True.
          If true, the data is log transformed.
    encoder_layer_size: 'tuple' or 'list'. Default: [128].
           Number of neurons in the encoder layers.
    dropout_rate: `float`. Default: 0.1.
           Probability of dropout in the bottleneck layer.
    epochs: 'int'. Default: 300.
           Number of total epochs.
    adversarial_weight: 'float'. Default: 0.6.
           Loss weight for the adversary. Controls the trade-off between
           reconstruction and batch correction.
    reg: 'float'. Default: 0.000.
           l2 kernel regularizer coefficient.
    batch_size: 'int'. Default: 32.
           Batch size for training.
    verbose: 'bool'. Default: False.
           If true, prints training information.
    es: 'int'. Default: 30.
           Patience for EarlyStopping.
    lr: 'int'. Default: 15.
           Patience for ReduceLROnPlateau.

    Returns:
    ---------
    {
        'imp': numpy matrix of imputed counts,
        'model': list of trained models,
        'loss_history': list of loss histories for each model
    }

    """
    if len(batch_info) == 0:
        raise ValueError("`batch_info` cannot be empty. Batch labels are required for adversarial training.")

    t1 = time.time()
    AE = AdversarialAutoencoder()
    data = data.astype('float32')

    data = take_norm(data, cellwise_norm=cellwise_norm, log1p=log1p)

    AE.set_input_data(data)
    AE.set_dropout_rate(dropout_rate)
    AE.set_epochs(epochs)
    AE.set_encoder_layer_size(encoder_layer_size)
    AE.set_batch_size(batch_size)
    AE.set_verbose(verbose)
    AE.set_early_stopping(es)
    AE.set_reduce_lr(lr)
    AE.set_reg(reg)
    AE.set_adversarial_weight(adversarial_weight)
    AE.set_batch_info(batch_info)

    nsamples = AE.nsamples
    ngene = AE.ngene
    print('{} cells and {} genes'.format(nsamples, ngene))
    print('{} batches detected'.format(AE.n_batches))

    models = []
    loss_history = []
    imps = np.zeros((nsamples, ngene))

    print('run the model 3 times and average the final imputation results for stability')
    for n in range(3):
        print('n_run = {}...'.format(n + 1))
        AE.create_model()
        AE.run_model()
        models.append(AE.model)
        loss_history.append(AE.his.history)
        imps = AE.imp + imps
    imps = imps / 3

    print('escape time is: {}'.format(time.time() - t1))
    return {'imp': imps, 'model': models, 'loss_history': loss_history}


class AdversarialAutoencoder(object):
    def __init__(self):
        self.encoder_layer_size = [128]
        self.input_data = None
        self.nsamples = 0
        self.ngene = 0
        self.input_layer = None
        self.neck_layer = None
        self.decode_layer = None
        self.adversary_layer = None
        self.model = None
        self.dropout_rate = 0
        self.epochs = 300
        self.adversarial_weight = 0
        self.n_batches = 0
        self.batch_dummy_label = None
        self.imp = None
        self.reg = 0
        self.batch_size = 32
        self.his = None
        self.early_stopping = 30
        self.reduce_lr = 15
        self.verbose = 0

    def set_encoder_layer_size(self, size):
        self.encoder_layer_size = size

    def set_input_data(self, input_data):
        self.input_data = input_data.astype('float32')
        self.nsamples = input_data.shape[0]
        self.ngene = input_data.shape[1]

    def set_dropout_rate(self, dropout_rate):
        self.dropout_rate = dropout_rate

    def set_epochs(self, n):
        self.epochs = n

    def set_adversarial_weight(self, w):
        self.adversarial_weight = w

    def set_n_batches(self, K):
        self.n_batches = K

    def set_reg(self, reg):
        self.reg = reg

    def set_batch_info(self, batch_info):
        n = len(batch_info)
        assert n == self.nsamples, 'Length of batch_info should equal the number of samples'
        labelid = pd.factorize(batch_info)[0]
        self.batch_dummy_label = to_categorical(labelid)
        self.n_batches = len(np.unique(labelid))

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def set_early_stopping(self, es):
        self.early_stopping = es

    def set_reduce_lr(self, lr):
        self.reduce_lr = lr

    def set_verbose(self, verbose):
        self.verbose = verbose

    def create_model(self):
        self.input_layer = tf.keras.layers.Input(shape=(self.ngene,))
        mid_layer = self.input_layer
        len_layer = len(self.encoder_layer_size)

        # Encoder
        for i in range(len_layer - 1):
            l_size = self.encoder_layer_size[i]
            mid_layer = tf.keras.layers.Dense(l_size, activation=tf.nn.relu,
                                              kernel_regularizer=tf.keras.regularizers.l2(self.reg))(mid_layer)

        # Bottleneck
        mid_layer = tf.keras.layers.Dense(self.encoder_layer_size[-1], activation=tf.nn.relu,
                                          kernel_regularizer=tf.keras.regularizers.l2(self.reg))(mid_layer)
        if self.dropout_rate > 0:
            mid_layer = tf.keras.layers.Dropout(self.dropout_rate)(mid_layer)
        self.neck_layer = mid_layer

        # Decoder branch
        mid_layer_d = mid_layer
        for i in range(len_layer - 1):
            l_size = self.encoder_layer_size[-2 - i]
            mid_layer_d = tf.keras.layers.Dense(l_size, activation=tf.nn.relu,
                                                kernel_regularizer=tf.keras.regularizers.l2(self.reg))(mid_layer_d)
        self.decode_layer = tf.keras.layers.Dense(self.ngene, activation=tf.nn.softplus, name='reconstruction',
                                                  kernel_regularizer=tf.keras.regularizers.l2(self.reg))(mid_layer_d)

        # Adversary branch
        # The Gradient Reversal Layer is inserted here
        adversary_input = GradientReversal()(self.neck_layer)
        self.adversary_layer = tf.keras.layers.Dense(self.n_batches, activation=tf.nn.softmax,
                                                     name='adversary',
                                                     kernel_regularizer=tf.keras.regularizers.l2(self.reg))(adversary_input)

        self.model = tf.keras.Model(inputs=self.input_layer,
                                    outputs=[self.adversary_layer, self.decode_layer])

    def run_model(self):
        CallBacks = []
        if self.early_stopping:
            es_cb = EarlyStopping(monitor='val_loss', patience=self.early_stopping, verbose=self.verbose)
            CallBacks.append(es_cb)
        if self.reduce_lr:
            lr_cb = ReduceLROnPlateau(monitor='val_loss', patience=self.reduce_lr, verbose=self.verbose)
            CallBacks.append(lr_cb)

        reconstruction_weight = 1 - self.adversarial_weight
        self.model.compile(loss={'adversary': 'categorical_crossentropy',
                                 'reconstruction': 'mean_squared_error'},
                           loss_weights={'adversary': self.adversarial_weight,
                                         'reconstruction': reconstruction_weight},
                           optimizer=tf.keras.optimizers.Adam())

        self.his = self.model.fit(self.input_data,
                                  {'adversary': self.batch_dummy_label,
                                   'reconstruction': self.input_data},
                                  batch_size=self.batch_size,
                                  verbose=self.verbose,
                                  epochs=self.epochs,
                                  validation_split=0.1,
                                  callbacks=CallBacks,
                                  shuffle=True)

        self.imp = self.model.predict(self.input_data)[1]
