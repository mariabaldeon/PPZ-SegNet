# PPZ-SegNet
In this work, we present PPZ-SegNet, a multi-object deep convolutional neural network ensemble for prostate segmentation. PPZ-SegNet is composed of a two-path 2D CNN and 3D CNN automatically constrcuted using a Bayesian hyperparameter optimization approach. The general structure of the networks is shown in Figure 1. The networks are composed of a down-sampling path followed by two up-sampling paths, denoted as up-pg (for prostate gland) and up-pz (for PZ), which produce the whole prostate and peripheral zone segmentation. 

![alt text](https://github.com/mariabaldeon/PPZ-SegNet/blob/main/imges/PPZNetStructure.jpg)

# Requirements
* Python 3.7
* Numpy 1.21.5
* Keras 2.11.0
* Tensorflow 2.11.0
* Simpleitk 2.1.1.2
* hdf5storage 0.1.18

# Training 
To carry out the training run: 
```
nohup python3 main.py --task train & 
```
The training will be performed in the five folds. For each fold two folders named 2d_training_logs and 3d_training_logs will appear. Insided the training logs and weights wil be saved for the 2d and 3d CNNs. 

# Evaluation
To carry out the evaluation run: 
```
nohup python3 main.py --task evaluate & 
```
The code will evaluate the 2D CNN ensemble, 3D CNN ensemble, and PPZSeg-Net. Evaluation metrics will be saved in a new folder named Evaluation_metrics. The segmentation results will be saved in the folders Results_2D.mat, Results_3D.mat, and Results_PPZSegNet.mat for the 2D CNN ensemble, 3D CNN ensemble, and PPZSeg-Net, respectively. 
