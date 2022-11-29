#!/usr/bin/env python
# coding: utf-8
import numpy as np
from keras.models import Model
from keras import initializers
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Add, Activation, SpatialDropout2D, BatchNormalization, Concatenate
from keras import backend as K
import math

def frst_blck(inp, nfilter, k1, k2, k3, actvfc):
    #print('input shape', inp.shape)
    x1=Conv2D(filters=nfilter, kernel_size=(k1,k1), padding='same', kernel_initializer='he_uniform')(inp)
    x1=BatchNormalization()(x1)
    x1= Activation(actvfc)(x1)

    x2=Conv2D(filters=nfilter, kernel_size=(k2,k2), padding='same', kernel_initializer='he_uniform')(x1)
    x2=BatchNormalization()(x2)
    x2= Activation(actvfc)(x2)

    x3=Conv2D(filters=nfilter, kernel_size=(k3,k3), padding='same', kernel_initializer='he_uniform')(x2)
    x3=BatchNormalization()(x3)
    x3= Activation(actvfc)(x3)
    #print('input shape before add', x3.shape)

    res= Add()([x1, x3])
    #print('input after add', res.shape)
    return res

def last_blck(frst_block, previous_block_wp, previous_block_pz, nfilter, k1, k2, k3, actvfc, p, add):
    #print('frst_block.shape ', frst_block.shape)
    #print('previous_block.shape ', previous_block.shape)

    #Segment PZ
    previous_block_pz=UpSampling2D(size=(2,2))(previous_block_pz)
    previous_block_pz=Conv2D(filters=nfilter, kernel_size=(2,2), padding='same', activation=actvfc,
                          kernel_initializer='he_uniform')(previous_block_pz)
    if add:
        x=Add()([previous_block_pz, frst_block])
    else:
        x=Concatenate()([previous_block_pz, frst_block])

    x= SpatialDropout2D(p)(x)

    x1=Conv2D(filters=nfilter, kernel_size=(k1,k1), padding='same', kernel_initializer='he_uniform')(x)
    x1=BatchNormalization()(x1)
    x1= Activation(actvfc)(x1)

    x2=Conv2D(filters=nfilter, kernel_size=(k2,k2), padding='same', kernel_initializer='he_uniform')(x1)
    x2=BatchNormalization()(x2)
    x2= Activation(actvfc)(x2)

    x3=Conv2D(filters=nfilter, kernel_size=(k3,k3), padding='same', kernel_initializer='he_uniform')(x2)
    x3=BatchNormalization()(x3)
    x3= Activation(actvfc)(x3)

    #PZ segmentation
    res= Add()([x1, x3])
    outputPZ= Conv2D(filters=1, kernel_size=1, activation='sigmoid', kernel_initializer='he_uniform', name="PZ" )(res)


    #Segmentation Prostate
    previous_block_wp=UpSampling2D(size=(2,2))(previous_block_wp)
    previous_block_wp=Conv2D(filters=nfilter, kernel_size=(2,2), padding='same', activation=actvfc,
                          kernel_initializer='he_uniform')(previous_block_wp)
    if add:
        x=Add()([previous_block_wp, frst_block])
    else:
        x=Concatenate()([previous_block_wp, frst_block])

    x= SpatialDropout2D(p)(x)
    x4=Conv2D(filters=nfilter, kernel_size=(k1,k1), padding='same', kernel_initializer='he_uniform')(x)
    x4=BatchNormalization()(x4)
    x4= Activation(actvfc)(x4)

    x5=Conv2D(filters=nfilter, kernel_size=(k2,k2), padding='same', kernel_initializer='he_uniform')(x4)
    x5=BatchNormalization()(x5)
    x5= Activation(actvfc)(x5)

    x6=Conv2D(filters=nfilter, kernel_size=(k3,k3), padding='same', kernel_initializer='he_uniform')(x5)
    x6=BatchNormalization()(x6)
    x6= Activation(actvfc)(x6)

    res2= Add()([x4, x6])
    res2=Concatenate()([res2, res])
    x7=Conv2D(filters=nfilter, kernel_size=(k3,k3), padding='same', kernel_initializer='he_uniform')(res2)
    x7=BatchNormalization()(x7)
    x7= Activation(actvfc)(x7)
    output= Conv2D(filters=1, kernel_size=1, activation='sigmoid', kernel_initializer='he_uniform', name="main" )(x7)
    #print('last after add', output.shape)
    return output, outputPZ

def res_downsampling(previous_block, nfilter, k1, k2, k3, actvfc, p):

    #print('previous_block.shape ', previous_block.shape)
    x= MaxPooling2D(pool_size=(2,2), strides=(2,2))(previous_block)
    x= SpatialDropout2D(p)(x)

    x1=Conv2D(filters=nfilter, kernel_size=(k1,k1), padding='same', kernel_initializer='he_uniform')(x)
    x1=BatchNormalization()(x1)
    x1= Activation(actvfc)(x1)

    x2=Conv2D(filters=nfilter, kernel_size=(k2,k2), padding='same', kernel_initializer='he_uniform')(x1)
    x2=BatchNormalization()(x2)
    x2= Activation(actvfc)(x2)

    x3=Conv2D(filters=nfilter, kernel_size=(k3,k3), padding='same', kernel_initializer='he_uniform')(x2)
    x3=BatchNormalization()(x3)
    x3= Activation(actvfc)(x3)
    #print('downsampling shape before add', x3.shape)

    res= Add()([x1, x3])
    #print('downsampling shape after add', res.shape)
    return res

def res_upsampling(downsampling_block, previous_block_wp,previous_block_pz, nfilter, k1, k2, k3, actvfc, p, add):

    #print('downsampling_block.shape ', downsampling_block.shape)
    #print('previous_block.shape ', previous_block.shape)

    # whole prostate
    previous_block_wp=UpSampling2D(size=(2,2))(previous_block_wp)
    previous_block_wp=Conv2D(filters=nfilter, kernel_size=(2,2), padding='same', activation=actvfc,
                          kernel_initializer='he_uniform')(previous_block_wp)

    if add:
        x=Add()([previous_block_wp, downsampling_block])
    else:
        x=Concatenate()([previous_block_wp, downsampling_block])

    x= SpatialDropout2D(p)(x)
    x1=Conv2D(filters=nfilter, kernel_size=(k1,k1), padding='same', kernel_initializer='he_uniform')(x)
    x1=BatchNormalization()(x1)
    x1= Activation(actvfc)(x1)

    x2=Conv2D(filters=nfilter, kernel_size=(k2,k2), padding='same', kernel_initializer='he_uniform')(x1)
    x2=BatchNormalization()(x2)
    x2= Activation(actvfc)(x2)

    x3=Conv2D(filters=nfilter, kernel_size=(k3,k3), padding='same', kernel_initializer='he_uniform')(x2)
    x3=BatchNormalization()(x3)
    x3= Activation(actvfc)(x3)
    #print('upsampling shape before add', x3.shape)
    res_wp= Add()([x1, x3])
    #print('upsampling shape before add', res.shape)

    # PZ
    previous_block_pz=UpSampling2D(size=(2,2))(previous_block_pz)
    previous_block_pz=Conv2D(filters=nfilter, kernel_size=(2,2), padding='same', activation=actvfc,
                          kernel_initializer='he_uniform')(previous_block_pz)

    if add:
        x4=Add()([previous_block_pz, downsampling_block])
    else:
        x4=Concatenate()([previous_block_pz, downsampling_block])

    x4= SpatialDropout2D(p)(x4)
    x4=Conv2D(filters=nfilter, kernel_size=(k1,k1), padding='same', kernel_initializer='he_uniform')(x4)
    x4=BatchNormalization()(x4)
    x4= Activation(actvfc)(x4)

    x5=Conv2D(filters=nfilter, kernel_size=(k2,k2), padding='same', kernel_initializer='he_uniform')(x4)
    x5=BatchNormalization()(x5)
    x5= Activation(actvfc)(x5)

    x6=Conv2D(filters=nfilter, kernel_size=(k3,k3), padding='same', kernel_initializer='he_uniform')(x5)
    x6=BatchNormalization()(x6)
    x6= Activation(actvfc)(x6)
    #print('upsampling shape before add', x3.shape)
    res_pz= Add()([x4, x6])
    #print('upsampling shape before add', res.shape)
    return res_wp, res_pz


# In[13]:


def get_2DCNN(h=128, w=128,p=0.5,k1=3,k2=3, k3=3, nfilter=32,actvfc='relu', blocks=9, channels=1, add=True):
    # Input_shape=(height, width, channels)
    inp=Input((h, w, channels))
    first_block=frst_blck(inp, nfilter, k1, k2, k3, actvfc)

    if blocks==3:

        down1=res_downsampling(first_block, nfilter*2, k1, k2, k3, actvfc, p)
        output, outputPZ= last_blck(first_block, down1,down1, nfilter, k1, k2, k3, actvfc, p, add)


    if blocks==5:
        down1=res_downsampling(first_block, nfilter*2, k1, k2, k3, actvfc, p)
        down2=res_downsampling(down1, nfilter*4, k1, k2, k3, actvfc, p)
        up3wp,up3pz =res_upsampling(down1, down2, down2,nfilter*2, k1, k2, k3, actvfc, p, add)
        output, outputPZ= last_blck(first_block, up3wp,up3pz, nfilter, k1, k2, k3, actvfc, p, add)

    if blocks==7:
        down1=res_downsampling(first_block, nfilter*2, k1, k2, k3, actvfc, p)
        down2=res_downsampling(down1, nfilter*4, k1, k2, k3, actvfc, p)
        down3=res_downsampling(down2, nfilter*8, k1, k2, k3, actvfc, p)
        up4wp, up4pz=res_upsampling(down2, down3, down3, nfilter*4, k1, k2, k3, actvfc, p, add)
        up5wp, up5pz=res_upsampling(down1, up4wp, up4pz, nfilter*2, k1, k2, k3, actvfc, p, add)
        output, outputPZ= last_blck(first_block, up5wp, up5pz, nfilter, k1, k2, k3, actvfc, p, add)

    if blocks==9:
        down1=res_downsampling(first_block, nfilter*2, k1, k2, k3, actvfc, p)
        down2=res_downsampling(down1, nfilter*4, k1, k2, k3, actvfc, p)
        down3=res_downsampling(down2, nfilter*8, k1, k2, k3, actvfc, p)
        down4=res_downsampling(down3, nfilter*16, k1, k2, k3, actvfc, p)
        up5wp, up5pz=res_upsampling(down3, down4, down4, nfilter*8, k1, k2, k3, actvfc, p, add)
        up6wp, up6pz=res_upsampling(down2, up5wp, up5pz, nfilter*4, k1, k2, k3, actvfc, p, add)
        up7wp, up7pz=res_upsampling(down1, up6wp, up6pz, nfilter*2, k1, k2, k3, actvfc, p, add)
        output, outputPZ= last_blck(first_block, up7wp, up7pz, nfilter, k1, k2, k3, actvfc, p, add)

    model= Model(inputs=inp, outputs=[output, outputPZ])
    return model

def prediction(kmodel, crpimg):
    imarr=np.array(crpimg).astype(np.float32)
    imarr = np.expand_dims(imarr, axis=0) #Adds a new dimension in the 0 axis that is the batch
    imarr, imarrpz = kmodel.predict(imarr)

    return imarr, imarrpz
