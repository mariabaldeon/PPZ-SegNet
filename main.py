#!/usr/bin/env python
# coding: utf-8

import argparse
import numpy as np
from DataPreprocessing.DataManager import DataManager, DataTraining, createImageFileList
from Train.Train_3DCNN import Train3D
from Train.Train_2DCNN import Train2D
from Networks.PPZSegNet import PPZSegNet
from Evaluation.EvaluationTest import Inference, Evaluate, createfolders

parser = argparse.ArgumentParser(prog="PPZ-SegNet")
parser.add_argument('--task', choices=['train', 'evaluate'], required=True, help='task to do: train PPZ-SegNet or evaluate' )
parser.add_argument('--dataTrain', type=str, default='./Datasets/Train', help='location of the training dataset')
parser.add_argument('--folds', type=int, default=5, help='Number of fold to create the ensemble')
parser.add_argument('--dataTest', type=str, default='./Datasets/Test', help='location of the testing dataset')
parser.add_argument('--imgSize', type=tuple, default=(256, 256, 23), help='standarized image size')
parser.add_argument('--dstRes', default=np.asarray([0.5,0.5,3],dtype=float), help='standarized resolution for the image in mm, in the x,y,z axis')
parser.add_argument('--patch_size', type=tuple, default=(128,128,23), help='image size for training the 3D CNN')
parser.add_argument('--batch_size2d', type=int, default=40, help='batch size for 2D CNN')
parser.add_argument('--batch_size3d', type=int, default=5, help='batch size for 3D CNN')
parser.add_argument('--mainloss', type=float, default=1, help='weight for the whole prostrate loss')
parser.add_argument('--lossPZ', type=float, default=0.1, help='weight for the PZ loss')
parser.add_argument('--num_epochs', type=int, default=2000, help='number of epochs to train CNN')
parser.add_argument('--dataaug3d', type=list, default=[4.9, 0.3, 0.2, 0.1, 0], help='data augmentation for 3D CNN: rotation, width shift, height shift, zoom shift, horizontal flip')
parser.add_argument('--dataaug2d', type=list, default=[33.3, 0.3, 0.4, 0.8, 1], help='data augmentation for 2D CNN: rotation, width shift, height shift, zoom shift, horizontal flip')
parser.add_argument('--gene3d', type=list, default=[[0.06,1,5,1,16,'elu',0.00001, 9]], help='gene to construct 3D CNN')
parser.add_argument('--gene2d', type=list, default=[[0.13,5,1,1,32,'relu',0.00001,9]], help='gene to construct 2D CNN')
parser.add_argument('--name2d', type=str, default="./Results_2D.mat", help='name of the folder to save predicted segmentations using 2D CNN')
parser.add_argument('--name3d', type=str, default="./Results_3D.mat", help='name of the folder to save predicted segmentations using 3D CNN')
parser.add_argument('--namePPZ', type=str, default="./Results_PPZSegNet.mat", help='name of the folder to save predicted segmentations using PPZSeg-Net')
args = parser.parse_args()

if args.task == 'train': 
    List, _=createImageFileList(args.dataTrain)
    for fold in range(1,args.folds+1): 
        # Obtain the training-validation set for each fold
        dta=DataTraining(List, fold)
        dta.getTrainingData()
        # Train 2D network
        train2DCNN=Train2D(dta.X_train2D, dta.y_train2D, dta.y_trainPZ2D, dta.X_val2D, dta.y_valPZ2D, dta.y_val2D, args.gene2d, args.num_epochs, args.batch_size2d, args.mainloss, args.lossPZ, args.dataaug2d)
        train2DCNN.run_training()
        # Train 3D network
        train3DCNN=Train3D(dta.X_train3D, dta.y_train3D, dta.y_trainPZ3D, dta.X_val3D, dta.y_valPZ3D, dta.y_val3D, args.gene3d, args.patch_size, args.num_epochs, args.batch_size3d, args.mainloss, args.lossPZ, args.dataaug3d)
        train3DCNN.run_training()


if args.task == 'evaluate': 
    List, _=createImageFileList(args.dataTest)
    createfolders(args.name2d, args.name3d, args.namePPZ)
    # Get the networks
    ppzsegnet= PPZSegNet(args.gene2d,  args.gene3d, args.imgSize, args.patch_size, args.folds)
    ensemble2DCNN, ensemble3DCNN=ppzsegnet.getPPZSegNet()
    
    for path in List: 
        # Get image to evaluate
        img=DataManager(path, args.imgSize)
        T2img,_,_ =img.get_img_gt()
        
        # Predict segmentation
        inf=Inference(T2img, ensemble2DCNN, ensemble3DCNN, args.imgSize)
        pred_finalWP2D, pred_finalPZ2D,pred_finalWP3D, pred_finalPZ3D,y_predWP,y_predPZ=inf.predictionAllEnsemble()

        # Write results for 2D ensemble 
        ypred2DWPitk,ypred2DPZitk =inf.writeResultsFromNumpyLabel(img.itkimg, pred_finalWP2D, pred_finalPZ2D, args.name2d, img.case)

        # Write results for 3D ensemble
        ypred3DWPitk,ypred3DPZitk =inf.writeResultsFromNumpyLabel(img.itkimg, pred_finalWP3D, pred_finalPZ3D, args.name3d, img.case)

        # Write results for PPZ-SegNet
        ypred2D3DWPitk,ypred2D3DPZitk =inf.writeResultsFromNumpyLabel(img.itkimg, y_predWP,y_predPZ, args.namePPZ, img.case)

        evaluate=Evaluate()
        evaluate.CalculateAllMetrics( [img.itkgtWP, img.itkgtPZ], [(ypred2DWPitk,ypred2DPZitk),(ypred3DWPitk,ypred3DPZitk),(ypred2D3DWPitk,ypred2D3DPZitk)], img.case)

