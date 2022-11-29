#!/usr/bin/env python
# coding: utf-8

import SimpleITK as sitk
import numpy as np
import os
import sys
from os import listdir
from os.path import isfile, join, splitext
import pandas as pd
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import math
import timeit
import re
import hdf5storage
from scipy.io import savemat
from Networks.CNN_3D import prediction

def createfolders(name2d, name3d, namePPZ):
    os.makedirs(name2d, exist_ok=True)
    os.makedirs(name3d, exist_ok=True)
    os.makedirs(namePPZ, exist_ok=True)


class Inference:
    """
    Receives the T2 image (T2img), 2D ensemble as a list,
    3D ensemble as a list, and produces the predicted segmentation
    for the whole prostate and PZ using the PPZ-SegNet,
    2D CNNs ensemble, 3D CNNs ensemble, and 2D-3D CNN ensembles
    """

    def __init__(self, T2img, ensemble2D, ensemble3D, imgSize,patch_size=(128,128,23),
                stride=(64,64,23), dstRes=np.asarray([0.5,0.5,3],dtype=float)):
        self.T2img = T2img
        self.ensemble2D= ensemble2D
        self.ensemble3D = ensemble3D
        self.imgSize=imgSize
        self.patch_size=patch_size
        self.stride=stride
        self.dstRes=dstRes
        assert len(self.ensemble2D)==len(self.ensemble3D)
        self.kfolds=len(self.ensemble2D)
        self.predictions2dWP=[]
        self.predictions3dWP=[]
        self.predictions2d3dWP=[]
        self.predictions2dPZ=[]
        self.predictions3dPZ=[]
        self.predictions2d3dPZ=[]

    def reshape(self, img):
        """
        Reshapes the image from shape (slices, height, width, channel)
        to (height, width, slices)
        """
        img=np.squeeze(img, axis=-1)
        img=img.transpose((1,2,0))
        return img

    def prediction_matrix2D(self,fold):
        """
        Predicts the segmentation using the 2D CNN
        trained in fold. Returns pz and wp prediction with shape (height, width, slices)
        Each pixel represents a probability of being part of the segmentation.
        """
        T2inp=np.squeeze(self.T2img, axis=0)
        T2inp=T2inp.transpose((2,0,1,3))
        y_pred_matrix=np.zeros(self.imgSize)
        # prediction recieves a matrix of shape (sl, row, col, channel)
        y_predWP, y_predPZ=self.ensemble2D[fold-1].predict(x=T2inp, batch_size=23)
        # returns size=(num,row,col,ch)
        y_predWP=self.reshape(y_predWP)
        y_predPZ=self.reshape(y_predPZ)

        return y_predWP, y_predPZ

    def num_patches(self, img_dim, patch_dim, stride):
        """
        Calculates the total number of patches in each dimension (width, height, slices)
        based on the strides defined, and the padding if the dimension is
        not perfectly divisible for the stride
        """
        n_patch=math.trunc(img_dim/stride)

        # If the dimension is perfectly divisible. No padding and perfectly computed number of patches
        if img_dim%stride==0:
            total_patches=n_patch
            lst_idx=(n_patch-1)*stride
            end_patch=lst_idx+patch_dim
            padding=end_patch-img_dim
            return total_patches, padding

        lst_idx=n_patch*stride
        end_patch=lst_idx+patch_dim
        padding=end_patch-img_dim
        total_patches=n_patch+1
        return total_patches, padding

    def prediction_matrix3D(self, fold):
        """
        Predicts the segmentation using the 3D CNN
        trained in fold. Returns pz and wp prediction with shape (height, width, slices)
        Each pixel represents a probability of being part of the segmentation.
        """
        num, row,col,sl, ch=self.T2img.shape
        pt_row, pt_col, pt_sl=self.patch_size
        str_row, str_col, str_sl=self.stride

        # total patches in each dimension and the padding at each dimension to have all strides
        num_row, pad_row=self.num_patches(row, pt_row, str_row)
        num_col, pad_col=self.num_patches(col, pt_col, str_col)
        num_sl, pad_sl=self.num_patches(sl, pt_sl, str_sl)

        X_pad=np.zeros((num, row+pad_row, col+pad_col, sl+pad_sl, ch))
        X_pad[:, pad_row:, pad_col:, pad_sl:,:]=self.T2img
        y_pred_matrixWP=np.zeros(X_pad.shape)
        y_pred_matrixPZ=np.zeros(X_pad.shape)
        # Counts the number of times the patch goes through the image to compute the average
        V=np.zeros(X_pad.shape)
        # for each patient
        for i in range(num):
            #For each row patch
            for j in range(num_row):
                #For each column patch
                for k in range(num_col):
                    #For each slice patch
                    for m in range(num_sl):
                        row_in=j*str_row
                        col_in=k*str_col
                        sl_in=m*str_sl
                        row_fin=row_in+pt_row
                        col_fin=col_in+pt_col
                        sl_fin=sl_in+pt_sl
                        Xi=X_pad[i,row_in:row_fin,col_in:col_fin,sl_in:sl_fin,:]
                        y_predWP, y_predPZ=prediction(self.ensemble3D[fold-1], Xi) #output size=(1,size,size,slices,1)
                        # Add previous predictions and the current prediction
                        y_pred_matrixWP[i,row_in:row_fin,col_in:col_fin,sl_in:sl_fin,:]=y_pred_matrixWP[i,row_in:row_fin,col_in:col_fin,sl_in:sl_fin,:]+y_predWP
                        y_pred_matrixPZ[i,row_in:row_fin,col_in:col_fin,sl_in:sl_fin,:]=y_pred_matrixPZ[i,row_in:row_fin,col_in:col_fin,sl_in:sl_fin,:]+y_predPZ
                       # Compute how many times a prediction has been dome to that pixel
                        Vi=np.zeros(X_pad.shape)
                        Vi[i,row_in:row_fin,col_in:col_fin,sl_in:sl_fin,:]=1.
                        V=V+Vi
        #compute the average of the predictions
        y_pred_matrixWP=np.true_divide(y_pred_matrixWP, V)
        y_pred_matrixPZ=np.true_divide(y_pred_matrixPZ, V)
        #take the padding out
        y_pred_matrixWP=np.squeeze(y_pred_matrixWP[:, pad_row:, pad_col:, pad_sl:,:])
        y_pred_matrixPZ=np.squeeze(y_pred_matrixPZ[:, pad_row:, pad_col:, pad_sl:,:])
        return y_pred_matrixWP, y_pred_matrixPZ


    def connected_component(self, y_pred):
        """
        Performs connected component analysis postprocessing.
        y_pred is the prediction with shape (height, width, slices)
        """
        yi=sitk.GetImageFromArray(y_pred)
        # Apply threshold
        thfilter=sitk.BinaryThresholdImageFilter()
        thfilter.SetInsideValue(1)
        thfilter.SetOutsideValue(0)
        thfilter.SetLowerThreshold(0.5)
        yi = thfilter.Execute(yi)
        # labels the objects in a binary image. Each distinct object is assigned a unique label
        # The final object labels start with 1 and are consecutive.
        # ObjectCount holds the number of connected components
        cc = sitk.ConnectedComponentImageFilter()
        yi = cc.Execute(sitk.Cast(yi,sitk.sitkUInt8))
        # Turn into a numpy array after the connected component analysis
        arrCC=np.transpose(sitk.GetArrayFromImage(yi).astype(dtype=float), [1, 2, 0])
        # Array of the length of all possible components in the image
        # If only 1 component is found (segments only 1 area) the array is size 1x2
        lab=np.zeros(int(np.max(arrCC)+1),dtype=float)
        # For each label after the connected component analysis
        for j in range(1,int(np.max(arrCC)+1)):
            # Add the number of pixels that have that label
            lab[j]=np.sum(arrCC==j)
        # The label that has the biggest number of segmented pixels
        activeLab=np.argmax(lab)
        # Keep only pixels of the image that have the activeLab label (label with most number of pixels)
        yi = (yi==activeLab)
        yi=sitk.GetArrayFromImage(yi).astype(dtype=float)
        return yi

    def prediction2DCNN(self, fold):
        """
        Predicts the whole prostate segmentation and pz segmention using the 2D
        model trained in fold and later applies a connected component
        post processing. Returns a .npy for each segmentation with
        shape (height, width, slices)

        """
        y_predWP, y_predPZ=self.prediction_matrix2D(fold)
        y_predWPcc=self.connected_component(y_predWP)
        y_predPZcc=self.connected_component(y_predPZ)
        return y_predWPcc, y_predPZcc

    def prediction3DCNN(self, fold):
        """
        Predicts the whole prostate segmentation and pz segmention using the 3D
        model trained in fold and later applies a connected component
        post processing. Returns a .npy for each segmentation with
        shape (height, width, slices)

        """
        y_predWP, y_predPZ=self.prediction_matrix3D(fold)
        y_predWPcc=self.connected_component(y_predWP)
        y_predPZcc=self.connected_component(y_predPZ)
        return y_predWPcc, y_predPZcc

    def prediction2D_3DEnsemble(self, fold):
        """
        Predicts the whole prostate segmentation and pz segmention
        by averaging the predictions of the 2D and 3D models trained
        in fold.
        Returns a .npy for each segmentation with
        shape (height, width, slices)
        """
        y_predWP2D, y_predPZ2D= self.prediction2DCNN(fold)
        y_predWP3D, y_predPZ3D=self.prediction3DCNN(fold)
        y_predWP=np.divide(y_predWP2D+y_predWP3D, 2.)
        y_predWP=self.connected_component(y_predWP)
        y_predPZ=np.divide(y_predPZ2D+y_predPZ3D, 2.)
        y_predPZ=self.connected_component(y_predPZ)

        return y_predWP, y_predPZ

    def prediction2DEnsemble(self):
        """
        Predicts the whole prostate segmentation and pz segmention
        by averaging the predictions of only the 2D CNNs
        Returns a .npy for each segmentation with
        shape (height, width, slices)
        """
        pred_finalWP=np.zeros(np.squeeze(self.T2img).shape)
        pred_finalPZ=np.zeros(np.squeeze(self.T2img).shape)
        for fold in range(1,self.kfolds+1):
            y_predWP, y_predPZ=self.prediction2DCNN(fold)
            y_predWP=self.connected_component(y_predWP)
            y_predPZ=self.connected_component(y_predPZ)
            self.predictions2dWP.append(y_predWP)
            self.predictions2dPZ.append(y_predPZ)
            pred_finalWP=np.add(pred_finalWP, y_predWP)
            pred_finalPZ=np.add(pred_finalPZ, y_predPZ)

        pred_finalWP=pred_finalWP/self.kfolds
        pred_finalWP=self.connected_component(pred_finalWP)
        pred_finalPZ=pred_finalPZ/self.kfolds
        pred_finalPZ=self.connected_component(pred_finalPZ)
        return pred_finalWP, pred_finalPZ

    def prediction3DEnsemble(self):
        """
        Predicts the whole prostate segmentation and pz segmention
        by averaging the predictions of only the 3D CNNs
        Returns a .npy for each segmentation with
        shape (height, width, slices)
        """
        pred_finalWP=np.zeros(np.squeeze(self.T2img).shape)
        pred_finalPZ=np.zeros(np.squeeze(self.T2img).shape)
        for fold in range(1,self.kfolds+1):
            y_predWP, y_predPZ=self.prediction3DCNN(fold)
            y_predWP=self.connected_component(y_predWP)
            y_predPZ=self.connected_component(y_predPZ)
            self.predictions3dWP.append(y_predWP)
            self.predictions3dPZ.append(y_predPZ)
            pred_finalWP=np.add(pred_finalWP, y_predWP)
            pred_finalPZ=np.add(pred_finalPZ, y_predPZ)

        pred_finalWP=pred_finalWP/self.kfolds
        pred_finalWP=self.connected_component(pred_finalWP)
        pred_finalPZ=pred_finalPZ/self.kfolds
        pred_finalPZ=self.connected_component(pred_finalPZ)
        return pred_finalWP, pred_finalPZ

    def predictionAllEnsemble(self):
        """
        Predicts the whole prostate segmentation and pz segmention, using the 2D ensembles,
        3D ensembles, and 2D-3DEnsembles.
        Returns the finals predictions
        by averaging the predictions of the 2D-3D CNNs
        Returns a .npy for each segmentation with
        shape (height, width, slices)
        """
        pred_finalWP2D, pred_finalPZ2D = self.prediction2DEnsemble()
        pred_finalWP3D, pred_finalPZ3D= self.prediction3DEnsemble()
        y_predWP=np.divide(pred_finalWP2D+pred_finalWP3D, 2.)
        y_predWP=self.connected_component(y_predWP)
        y_predPZ=np.divide(pred_finalPZ2D+pred_finalPZ3D, 2.)
        y_predPZ=self.connected_component(y_predPZ)

        for fold in range(self.kfolds):
            y_predWP2d3d=np.divide(self.predictions2dWP[fold]+self.predictions3dWP[fold], 2.)
            y_predWP2d3d=self.connected_component(y_predWP2d3d)
            self.predictions2d3dWP.append(y_predWP2d3d)
            y_predPZ2d3d=np.divide(self.predictions2dPZ[fold]+self.predictions3dPZ[fold], 2.)
            y_predPZ2d3d=self.connected_component(y_predPZ2d3d)
            self.predictions2d3dPZ.append(y_predPZ2d3d)

        return pred_finalWP2D, pred_finalPZ2D,pred_finalWP3D, pred_finalPZ3D,y_predWP,y_predPZ

    def transformToOrgResolution(self, img, result):
        """
        Transforms the segmentation (result) to the resolution of the original image (img)
        """
        # create an image where to copy the information
        toWrite=sitk.Image(img.GetSize()[0],img.GetSize()[1],img.GetSize()[2],sitk.sitkFloat32)
        factor = np.asarray(img.GetSpacing()) / self.dstRes
        factorSize = np.asarray(img.GetSize() * factor, dtype=float)
        newSize = np.max([factorSize, self.imgSize], axis=0)
        newSize = newSize.astype(dtype=int).tolist()
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(img)
        resampler.SetOutputSpacing(self.dstRes)
        resampler.SetSize(newSize)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        toWrite = resampler.Execute(toWrite)
        imgCentroid = np.asarray(newSize, dtype=float) / 2.0
        imgStartPx = (imgCentroid - np.asarray(self.imgSize,dtype=int) / 2.0).astype(dtype=int)
        for dstX, srcX in zip(range(0, result.shape[0]), range(imgStartPx[0],int(imgStartPx[0]+self.imgSize[0]))):
                for dstY, srcY in zip(range(0, result.shape[1]), range(imgStartPx[1], int(imgStartPx[1]+self.imgSize[1]))):
                    for dstZ, srcZ in zip(range(0, result.shape[2]), range(imgStartPx[2], int(imgStartPx[2]+self.imgSize[2]))):
                        try:
                            # The shape in itk=(column, row, slice) the shape in np.array (row, column, slice)
                            # The Index type reverses the order so that with Index[0] = col, Index[1] = row, Index[2] = slice
                            # Copy each pixel of the segmented result into the toWrite img. the segmentation is centered
                            toWrite.SetPixel(int(srcX),int(srcY),int(srcZ),float(result[dstY, dstX,dstZ]))
                        except:
                            pass
        resampler.SetOutputSpacing([img.GetSpacing()[0], img.GetSpacing()[1], img.GetSpacing()[2]])
        resampler.SetSize(img.GetSize())
        toWrite = resampler.Execute(toWrite)
        return toWrite

    def writeResultsFromNumpyLabel(self, img, predWZ, predPZ, resultsDirmat, case):

        toWriteWP=self.transformToOrgResolution(img, predWZ)
        toWritePZ=self.transformToOrgResolution(img, predPZ)
        imgnpWP=np.transpose(sitk.GetArrayFromImage(toWriteWP).astype(dtype=float), [1, 2, 0])
        imgnpPZ=np.transpose(sitk.GetArrayFromImage(toWritePZ).astype(dtype=float), [1, 2, 0])

        current_img = {"Prostate": imgnpWP, "PZ":imgnpPZ}
        # case='ProstateX-0001.mat'
        mat_file_name=str(resultsDirmat+"/"+case)
        hdf5storage.savemat(mat_file_name, current_img,format='7.3')

        return toWriteWP, toWritePZ


class Evaluate():
    """
    Calculates the dice and HD between the ground truth and
    predicted segmentation
    Accepts only itk images
    """
    # Class variable that stores the evaluation metrics of all images
    eval_metrics = pd.DataFrame(columns=['Case','2D-WP_DS','2D-WP_HD', '2D-PZ_DS','2D-PZ_HD', '3D-WP_DS','3D-WP_HD',
                                        '3D-PZ_DS','3D-PZ_HD','PPZSegNet-WP_DS','PPZSegNet-WP_HD',
                                       'PPZSegNet-PZ_DS','PPZSegNet-PZ_HD'])

    def calculateHD(self, gt, pred):
        """ Calculates the Hausdorff distance using
        itk images"""
        hausdorff_distance_filter=sitk.HausdorffDistanceImageFilter()
        hausdorff_distance_filter.Execute(gt,pred)
        return hausdorff_distance_filter.GetHausdorffDistance()

    def calculateDS(self, gt, pred):
        """ Calculates the Dice using
        itk images"""
        overlap_measure_filter=sitk.LabelOverlapMeasuresImageFilter()
        overlap_measure_filter.Execute(gt,pred)
        return overlap_measure_filter.GetDiceCoefficient()

    def CalculateMetrics(self, gt, pred):
        """ Calculates the Hausdorff and Dice using
        itk images"""
        gt=sitk.Cast(gt,sitk.sitkUInt8)
        pred=sitk.Cast(pred,sitk.sitkUInt8)
        hd=self.calculateHD(gt, pred)
        ds=self.calculateDS(gt, pred)
        return hd, ds

    def CalculateAllMetrics(self, y_true, y_pred, case):
        """ Calculates the Hausdorff and Dice using
        itk images for the 2D ensemble, 3D ensemble, and PPZ-SegNet.
        Saves the evaluation metrics in the EvaluationMetrics Folder"""
        os.makedirs('./Evaluation_Metrics', exist_ok=True)
        metrics=[]
        for pred in y_pred:
                hdWP, dsWP= self.CalculateMetrics(pred[0],y_true[0])
                hdPZ, dsPZ= self.CalculateMetrics(pred[1],y_true[1])
                metrics.extend([dsWP, hdWP, dsPZ, hdPZ])
        Evaluate.eval_metrics=Evaluate.eval_metrics.append({'Case': case ,'2D-WP_DS': metrics[0] ,
                                                            '2D-WP_HD': metrics[1], '2D-PZ_DS': metrics[2],'2D-PZ_HD': metrics[3], '3D-WP_DS':metrics[4], '3D-WP_HD': metrics[5],
                                                            '3D-PZ_DS':metrics[6],'3D-PZ_HD':metrics[7], 'PPZSegNet-WP_DS': metrics[8],'PPZSegNet-WP_HD': metrics[9],
                                                            'PPZSegNet-PZ_DS':metrics[10],'PPZSegNet-PZ_HD':metrics[11]}, ignore_index=True)
        Evaluate.eval_metrics.to_csv('./Evaluation_Metrics/EvaluationMetrics.csv')
