#!/usr/bin/env python
# coding: utf-8


import numpy as np
import os
import re
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from .CNN_3D import get_3DCNN
from .CNN_2D import get_2DCNN

#loss coeficients
smooth=0.5
threshold=0
def dice_coef(y_true, y_pred):
    y_true_f=K.flatten(y_true)
    y_pred_f=K.flatten(y_pred)
    intersection=K.sum(y_true_f*y_pred_f)
    return(2.*intersection+smooth)/((K.sum(y_true_f*y_true_f)) + K.sum(y_pred_f*y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1.-dice_coef(y_true, y_pred)

class PPZSegNet:
    """
    Creates the 2D-3D PPZSegNet
    kfolds: number of folds to create the 2D-3D ensembles
    gene2D: gene for the construction of the 2D CNN
    gene3D: gene for the construction of the 3D CNN
    imgSize: size of the images, we asume the 2D CNN is
    trained with slices of the same size
    patch_size: size of the patch for the 3D CNN
    """
    def __init__(self, gene2D,  gene3D, imgSize, patch_size, kfolds=5):
        self.kfolds= kfolds
        self.gene2D= gene2D
        self.gene3D =  gene3D
        self.imgSize = imgSize
        self.patch_size = patch_size
        self.weights=[]
        self.ensemble2D=[]
        self.ensemble3D=[]

    def getWeights(self,  basePath='Networks/weights'):
        """
        Uploads the weights of the 2D and 3D CNNs

        """
        try:
            for f in os.listdir(basePath):
                if ".hdf5" in f:
                    self.weights.append(os.path.join(basePath,f))
            print(self.weights)
        except FileNotFoundError:
            print("The path to the weights is incorrect")
        assert len(self.weights)==self.kfolds*2

    def get2DCNN(self,fold):
        """
        Gets the trained 2D CNN from the fold specified
        """
        weightName='k'+str(fold)+'_2D'
        weight2D= [x for x in self.weights if re.search(weightName, x)]
        print('weights2D ', weight2D)

        model2D= get_2DCNN(h=self.imgSize[0],w=self.imgSize[1],
            p=self.gene2D[0][0],k1=self.gene2D[0][1],k2=self.gene2D[0][2],
            k3=self.gene2D[0][3], nfilter=self.gene2D[0][4],
            actvfc=self.gene2D[0][5], blocks=self.gene2D[0][7],
            channels=1)
        model2D.load_weights(weight2D[0])
        adam=tf.keras.optimizers.Adam(learning_rate=self.gene2D[0][6],
                                      beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model2D.compile(loss=dice_coef_loss, optimizer=adam)
        return model2D

    def get3DCNN(self,fold):
        """
        Gets the trained 3D CNN from the fold specified
        """

        weightName='k'+str(fold)+'_3D'
        weight3D= [x for x in self.weights if re.search(weightName, x)]
        print('weights3D ', weight3D)

        model3D= get_3DCNN(h=self.patch_size[0], w=self.patch_size[1] ,
            p=self.gene3D[0][0], k1=self.gene3D[0][1],
            k2=self.gene3D[0][2], k3=self.gene3D[0][3],
            nfilter=self.gene3D[0][4],actvfc=self.gene3D[0][5],
                blocks=self.gene3D[0][7], slices=self.patch_size[2],
                channels=1)
        model3D.load_weights(weight3D[0])
        adam=tf.keras.optimizers.Adam(learning_rate=self.gene3D[0][6],
                        beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        model3D.compile(loss=dice_coef_loss, optimizer=adam)
        return model3D

    def getPPZSegNet(self):
        """
        Creates the 2D CNNs and 3D CNNs for the ensemble
        """
        self.getWeights()
        for fold in range(1,self.kfolds+1):
            self.ensemble2D.append(self.get2DCNN(fold))
            self.ensemble3D.append(self.get3DCNN(fold))

        return self.ensemble2D, self.ensemble3D
