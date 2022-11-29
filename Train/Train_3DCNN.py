#!/usr/bin/env python
# coding: utf-8

import numpy as np
import logging
from keras import optimizers
from keras.callbacks import CSVLogger,ModelCheckpoint,Callback
from keras import backend as K
from Networks.CNN_3D import get_3DCNN
import math
from .ImageGenerator_3dcrop import ImageDataGenerator
import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"


smooth=0.5
def dice_coef(y_true, y_pred):
    y_true_f=K.flatten(y_true)
    y_pred_f=K.flatten(y_pred)
    intersection=K.sum(y_true_f*y_pred_f)
    return(2.*intersection+smooth)/((K.sum(y_true_f*y_true_f)) + K.sum(y_pred_f*y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef(y_true, y_pred)

class Train3D():
    """Trains the 3D CNN. Each 3D image is cropped to shape patch_size.
    """
    def __init__(self, X_train, y_trainWP, y_trainPZ, X_val, y_valPZ, y_valWP, gene, patch_size,num_epochs, batch_size, mainloss, lossPZ, dataaug):
        self.gene= gene
        self.patch_size= patch_size
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
        self.X_val=self.val_convert_patch(X_val)
        self.y_valPZ=self.val_convert_patch(y_valPZ)
        self.y_valWP=self.val_convert_patch(y_valWP)
        self.datagenX=self.getGenerator()
        self.datagenY=self.getGenerator()
        self.datagenYPZ=self.getGenerator()

    def val_stride(self, img_dim, patch_dim):
        total_patch=math.ceil(img_dim/patch_dim)
        if total_patch==1:
            return img_dim, total_patch
        pix_dif=(patch_dim*total_patch)-img_dim
        stride_dif=math.ceil(pix_dif/(total_patch-1))
        stride=patch_dim-stride_dif
        return stride, total_patch

    def val_convert_patch(self, X_val):
        num, row, col, sl, ch= X_val.shape
        pt_row, pt_col, pt_sl= self.patch_size
        row_str, num_row=self.val_stride(row, pt_row)
        col_str, num_col=self.val_stride(col, pt_col)
        sl_str, num_sl=self.val_stride(sl, pt_sl)
        img_patch=num_row*num_col*num_sl
        total_patch=num*img_patch
        X_val_patch=np.zeros((total_patch, pt_row, pt_col, pt_sl, ch))
        ix_patch=0
        for i in range(num):
            for j in range(num_row):
                for k in range(num_col):
                    for m in range(num_sl):
                        row_in=j*row_str
                        col_in=k*col_str
                        sl_in=m*sl_str
                        row_fin=row_in+pt_row
                        col_fin=col_in+pt_col
                        sl_fin=sl_in+pt_sl
                        X_val_patch[ix_patch,:,:,:,0]=X_val[i,row_in:row_fin,col_in:col_fin,sl_in:sl_fin,0]
                        ix_patch=ix_patch+1
        return X_val_patch

    def getGenerator(self):
        return ImageDataGenerator(rotation_range=self.rot, width_shift_range=self.widths, height_shift_range=self.heights, zoom_range=self.zooms,
                                      horizontal_flip=self.hflip, data_format='channels_last', random_crop=self.patch_size)

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
        os.makedirs('3d_training_logs', exist_ok=True)
        model= get_3DCNN(h=self.patch_size[0],w=self.patch_size[1], p=self.gene[0][0],k1=self.gene[0][1],k2=self.gene[0][2], k3=self.gene[0][3], nfilter=self.gene[0][4],actvfc=self.gene[0][5],
                   blocks=self.gene[0][7], slices=self.patch_size[2], channels=1)
        adam=optimizers.Adam(lr=self.gene[0][6], beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model.compile(loss=[dice_coef_loss,dice_coef_loss], optimizer=adam, loss_weights= [self.mainloss, self.lossPZ], metrics=[dice_coef])
        csv_logger = CSVLogger('3d_training_logs/3d_training.log')
        model_check=ModelCheckpoint(filepath= "3d_training_logs/weights.{epoch:02d}-{val_main_dice_coef:.2f}.hdf5" , monitor='val_main_loss', verbose=0, save_best_only=True)
        history=model.fit_generator(self.generator(self.X_train, self.y_trainWP, self.y_trainPZ),
                                steps_per_epoch=(self.X_train.shape[0]/self.batch_size),
                validation_data=self.next_batch(self.X_val, self.y_valWP, self.y_valPZ, 1), validation_steps=(self.X_val.shape[0]), epochs=self.num_epochs,
                            callbacks=[csv_logger, model_check])
