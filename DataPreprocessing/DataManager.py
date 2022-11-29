#!/usr/bin/env python
# coding: utf-8

import SimpleITK as sitk
import numpy as np
import os
from os import listdir
from os.path import isfile, join, splitext
import hdf5storage
from scipy.io import savemat
import random
import re
from .preprocessing import del_out_3D, norm_max_3D
from math import ceil

def createImageFileList(srcFolder):
    """
    Use:
        The function structures lists with complete paths (based on srcFolder) and list only with names of all files
    Parameters:
        srcFolder: directory defined above where .mat images are located
    Returns:
        List: a list consisting of full path of each image, and the individual image names. e.g. ['../ProstateSegmentation/Images/ProstateX-0057.mat','../ProstateSegmentation/Images/ProstateX-0043.mat',...]
        img_name: the raw names of each image files .mat e.g. ['ProstateX-0057.mat','ProstateX-0043.mat',...]
        Outputs will be used in loadTrainingData()
    """
    List = [join(srcFolder, f) for f in listdir(srcFolder)] #join() takes as input iterable, in this case it joins the source path with each image name
    img_name=[]
    for path in List:
        if path[-4:]=='.mat': #Only appends name if is a .mat file
            name=path[-18:] #[-18:] Each file name is in the last part of dir. Each .mat name has exactly 18 characters. Check and update if neccesary
            img_name.append(name)
        else:
            pass
    return List, img_name

class DataManager:
    """
    Reads a .mat image and returns the image,
    whole prostate segmetation, and PZ segmentation
    after pre processing
    path: path to the image to read
    imgSize: normalized 3D size for all images
    dstRes: normalized image resolution
    """
    def __init__(self, path, imgSize, dstRes=np.asarray([0.5,0.5,3],dtype=float)):
        self.path=path
        self.case=path[-18:]
        self.image=self.readImage()
        self.imgSize= imgSize
        self.dstRes = dstRes
        self.rescalFilt=sitk.RescaleIntensityImageFilter()
        self.rescalFilt.SetOutputMaximum(1)
        self.rescalFilt.SetOutputMinimum(0)

    def readImage(self):
        """
        Reads the .mat image and saves resolution information
        """
        image=hdf5storage.loadmat(self.path)
        self.y_mm=image['Parameters']['PixelSpacing'][0][0][1] #Get 2nd value of pixelSpacing
        self.x_mm=image['Parameters']['PixelSpacing'][0][0][0] #Get 1st value of pixelSpacing
        self.z_mm=image['Parameters']['SliceThickness'][0][0][0]
        return image

    def getImage(self):
        """
        Returns the images in .npy format
        """
        T2img=self.image["T2"]
        T2img = np.transpose(T2img, (2,0,1)) #Set correct formating to use as ITK Image
        itkimg = sitk.GetImageFromArray(T2img)
        itkimg.SetSpacing((self.x_mm,self.y_mm,self.z_mm))
        self.itkimg=self.rescalFilt.Execute(sitk.Cast(itkimg,sitk.sitkFloat32))
        return self.getNumpyData(self.itkimg, sitk.sitkBSpline)

    def getWPGT(self):
        """
        Returns the whole prostate segmentation in .npy format
        """
        WPseg=self.image["Prostate"]
        WPseg=WPseg*1.0 #Since it is a True/False matrix, *1.0 transforms to numerical values as in T2 format
        WPseg = np.transpose(WPseg, (2,0,1)) #Set correct formating to use as ITK Image
        itkgt = sitk.GetImageFromArray(WPseg)
        itkgt.SetSpacing((self.x_mm,self.y_mm,self.z_mm))
        self.itkgtWP=sitk.Cast(itkgt,sitk.sitkFloat32)
        return self.getNumpyData(self.itkgtWP, sitk.sitkNearestNeighbor)

    def getPZGT(self):
        """
        Returns the PZ prostate segmentation in .npy format
        """
        PZseg=self.image["PZ"]
        PZseg=PZseg*1.0 #Since it is a True/False matrix, *1.0 transforms to numerical values as in T2 format
        PZseg = np.transpose(PZseg, (2,0,1)) #Set correct formating to use as ITK Image
        itkgt = sitk.GetImageFromArray(PZseg)
        itkgt.SetSpacing((self.x_mm,self.y_mm,self.z_mm))
        self.itkgtPZ=sitk.Cast(itkgt,sitk.sitkFloat32)
        return self.getNumpyData(self.itkgtPZ, sitk.sitkNearestNeighbor)

    def getNumpyData(self, img, method):
        """
        Use:
            Uses the ITK images and normalizes in size and resolution
            Returns np arrays
        Parameters:
            img: ITK image
            method: interpolation method to standarize size
        Returns:
            npimg: image in a numpy array
        """
        factor = np.asarray(img.GetSpacing()) / self.dstRes
        factorSize = np.asarray(img.GetSize() * factor, dtype=float)
        # The new size of the image considering the spacing
        newSize = np.max([factorSize, self.imgSize], axis=0)
        newSize = newSize.astype(dtype=int).tolist()
        # Filter used to resample an existing image through a coordinate transform and interpolating function
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(img)
        # Set the spacing of the output image
        resampler.SetOutputSpacing(self.dstRes)
        # Set the Size of the output image
        resampler.SetSize(newSize)
        # Set a linear interpolation method
        resampler.SetInterpolator(method)
        imgResampled = resampler.Execute(img)
        # Set the centroid to be in the center of the image (in the middle of the newsize)
        imgCentroid = np.asarray(newSize, dtype=float) / 2.0
        # Offset between the centroid in the resampled image and the centroid with the Volume Size defined
        imgStartPx = (imgCentroid - np.asarray(self.imgSize,dtype=int) / 2.0).astype(dtype=int)
        regionExtractor = sitk.RegionOfInterestImageFilter()
        #Size in pixels of the region extracted
        regionExtractor.SetSize(list(self.imgSize))
        # Sets the inclusive starting index of the region extracted
        regionExtractor.SetIndex(imgStartPx.tolist())
        imgResampledCropped = regionExtractor.Execute(imgResampled)

        # Transpose to have rows, columns, slices
        npimg = np.transpose(sitk.GetArrayFromImage(imgResampledCropped).astype(dtype=float), [1, 2, 0])
        return npimg

    def preprocessing(self, npimg):
        """
        Preprocess each image by subtracting the mean and
        dividing by std. Add channel and batch dimension.
        Substitute outliers, and put in range between 0-1
        """
        mean = np.mean(npimg[npimg>0])
        std = np.std(npimg[npimg>0])
        npimg-=mean
        npimg/=std
        npimg=np.expand_dims(npimg, axis=-1)
        npimg=np.expand_dims(npimg, axis=0)
        npimg=del_out_3D(npimg, 3)
        npimg=norm_max_3D(npimg)

        return npimg

    def get_img_gt(self):
        """
        Reads the images/gt, transforms to .npy format, and does
        all preprocessing. Returns T2 image, whole prostate segmentation
        and PZ segmentation
        """
        T2img=self.getImage()
        T2img=self.preprocessing(T2img)
        gtpz=self.getPZGT()
        gtwp=self.getWPGT()

        return T2img, gtwp, gtpz

class DataTraining():
    """
    Creates the training and validation set for the fold.
    Returns the training matrix in .npy format
    pathlist=list with the path to all training images
    fold=number of fold

    """
    def __init__(self, pathlist, fold, imgSize=(256, 256, 23), dstRes=np.asarray([0.5,0.5,3],dtype=float)):
        self.pathlist=pathlist
        self.fold=fold
        self.num_cases=len(self.pathlist)
        self.val_num=ceil(self.num_cases*0.20)
        self.train_num=self.num_cases-self.val_num
        self.imgSize=imgSize
        self.dstRes=dstRes
        if self.val_num==0:
            raise Exception("Validation set should have at least one case")
        if self.num_cases==0:
            raise Exception("Training set should have at least one case")

    def CreateFold(self):
        """
        Creates a list with the cases for training and cases for validation
        randonmly selecting 20% cases for validation and 80% for training
        """
        all_cases=list(range(0,self.num_cases))
        val_list=np.random.RandomState(seed=0).choice(all_cases, size=self.num_cases,replace=False)
        start_fold=(self.fold-1)*self.val_num
        end_fold=start_fold+self.val_num
        print(start_fold)
        print(end_fold)
        val_fold=val_list[start_fold:end_fold]
        train_fold=[e for e in all_cases if e not in val_fold]
        train_fold=np.sort(train_fold)
        val_fold=np.sort(val_fold)
        return train_fold, val_fold

    def find_path(self, case_num):
        """ Finds the path to the case that have has case_num
        Used to create validation and training matrix"""
        if case_num<10:
            pat='ProstateX-000'+str(case_num)
        elif case_num>=10 and i<100:
            pat='ProstateX-00'+str(case_num)
        elif case_num>=100:
            pat='ProstateX-0'+str(case_num)
        print(pat)
        pat_path = [x for x in self.pathlist if re.search(pat, x)]
        # If len not equal to 1 you haven´t found a case with that number or you have more cases with that number
        assert len(pat_path)==1
        return pat_path[0]

    def CreateSet3D(self, list_fold):
        """
        Creates the 2D training set based on the cases selected in list_fold
        """
        array_size=(len(list_fold),)+(self.imgSize)+(1,)
        X_train=np.zeros(array_size)
        y_train=np.zeros(array_size)
        y_trainPZ=np.zeros(array_size)
        for i, pat in enumerate(list_fold):
            pat_path=self.find_path(pat)
            dataimport=DataManager(pat_path, self.imgSize, self.dstRes)
            T2img, gtwp, gtpz =dataimport.get_img_gt()
            X_train[i]=T2img[0]
            y_train[i,:,:,:,0]=gtwp
            y_trainPZ[i,:,:,:,0]=gtpz
        return X_train, y_train,y_trainPZ

    def CreateSet2D(self, list_fold):
        """
        Creates the 2D training set based on the cases selected in list_fold
        """
        array_size=(len(list_fold)*self.imgSize[-1],)+(self.imgSize[:2])+(1,)
        X_train=np.zeros(array_size)
        y_train=np.zeros(array_size)
        y_trainPZ=np.zeros(array_size)
        for i, pat in enumerate(list_fold):
            start=i*self.imgSize[-1]
            end=start+self.imgSize[-1]
            pat_path=self.find_path(pat)
            dataimport=DataManager(pat_path, self.imgSize, self.dstRes)
            T2img, gtwp, gtpz =dataimport.get_img_gt()
            T2img=np.squeeze(T2img)
            T2img=T2img.transpose(2,0,1)
            gtwp=gtwp.transpose(2,0,1)
            gtpz=gtpz.transpose(2,0,1)
            X_train[start:end,:,:,0]=T2img
            y_train[start:end,:,:,0]=gtwp
            y_trainPZ[start:end,:,:,0]=gtpz
        return X_train, y_train,y_trainPZ

    def getTrainingData(self):
        """
        Saves the 2D and 3D npy matrix for training as attributes
        """
        train_fold, val_fold=self.CreateFold()
        self.X_train2D, self.y_train2D,self.y_trainPZ2D=self.CreateSet2D(train_fold)
        self.X_val2D, self.y_val2D,self.y_valPZ2D=self.CreateSet2D(val_fold)
        self.X_train3D, self.y_train3D,self.y_trainPZ3D=self.CreateSet3D(train_fold)
        self.X_val3D, self.y_val3D,self.y_valPZ3D=self.CreateSet3D(val_fold)
