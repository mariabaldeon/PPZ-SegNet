#!/usr/bin/env python
# coding: utf-8

import numpy as np
import logging
from keras import optimizers
from keras.callbacks import CSVLogger,ModelCheckpoint,Callback
from keras import backend as K
from Networks.CNN_2D import get_2DCNN
from .Train_3DCNN import dice_coef, dice_coef_loss
from keras.preprocessing.image import ImageDataGenerator
import math
import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"

class Train2D():
    """Trains the 2D CNN with slices from the 3D images.
    """
    def __init__(self, X_train, y_trainWP, y_trainPZ, X_val, y_valPZ, y_valWP, gene,num_epochs, batch_size, mainloss, lossPZ, dataaug):
        self.gene= gene
        self.num_epochs= num_epochs
        self.batch_size=batch_size
        self.mainloss=mainloss
        self.lossPZ= lossPZ
        self.rot= dataaug[0]
        self.widths=dataaug[1]
        self.heights=dataaug[2]
        self.zooms=dataaug[3]
        self.hflip=dataaug[4]
        self.X_train=X_train
        self.y_trainWP=y_trainWP
        self.y_trainPZ=y_trainPZ
        self.X_val=X_val
        self.y_valPZ=y_valPZ
        self.y_valWP=y_valWP
        self.datagenX=self.getGenerator()
        self.datagenY=self.getGenerator()
        self.datagenYPZ=self.getGenerator()

    def getGenerator(self):
        return ImageDataGenerator(rotation_range=self.rot, width_shift_range=self.widths, height_shift_range=self.heights, zoom_range=self.zooms,
                                      horizontal_flip=self.hflip, data_format='channels_last')

    def generator(self, x, y, ypz):
        genX = self.datagenX.flow(x, batch_size=self.batch_size, seed=7)
        genY = self.datagenY.flow(y, batch_size=self.batch_size, seed=7)
        genYPZ = self.datagenYPZ.flow(ypz, batch_size=self.batch_size, seed=7)
        while True:
            xi = genX.next()
            yi = genY.next()
            ypzi = genYPZ.next()
            yield xi, [yi, ypzi]

    def next_batch(self, Xs, ys, yp, size):
        while True:
            perm=np.random.permutation(Xs.shape[0]) #changes the order of the samples
            for i in np.arange(0, Xs.shape[0], size):
                X=Xs[perm[i:i+size]]
                y=ys[perm[i:i+size]]
                y2=yp[perm[i:i+size]]
                yield X, [y, y2]

    def run_training(self):
        os.makedirs('2d_training_logs', exist_ok=True)
        model= get_2DCNN(h=self.X_train.shape[1],w=self.X_train.shape[2], p=self.gene[0][0],k1=self.gene[0][1],k2=self.gene[0][2], k3=self.gene[0][3], nfilter=self.gene[0][4],actvfc=self.gene[0][5],
                   blocks=self.gene[0][7], channels=1)
        adam=optimizers.Adam(lr=self.gene[0][6], beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(loss=[dice_coef_loss,dice_coef_loss], optimizer=adam, loss_weights= [self.mainloss, self.lossPZ], metrics=[dice_coef])
        csv_logger = CSVLogger('2d_training_logs/2d_training.log')
        model_check=ModelCheckpoint(filepath= "2d_training_logs/weights.{epoch:02d}-{val_main_dice_coef:.2f}.hdf5" , monitor='val_main_loss', verbose=0, save_best_only=True)
        history=model.fit_generator(self.generator(self.X_train, self.y_trainWP, self.y_trainPZ),
                                steps_per_epoch=(self.X_train.shape[0]/self.batch_size),
                validation_data=self.next_batch(self.X_val, self.y_valWP, self.y_valPZ, 1), validation_steps=(self.X_val.shape[0]), epochs=self.num_epochs,
                            callbacks=[csv_logger, model_check])
