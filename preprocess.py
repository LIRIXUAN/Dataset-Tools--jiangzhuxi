import os
import json
import cv2
import zipfile
import numpy as np
import random
from progressbar import ProgressBar

def add_noise(img,min_std = 10, max_std=0 ,min_mean = 0,max_mean = 0):
    std = random.randint(min_std,max_std)
    mean = random.randint(min_mean,max_mean)
    img = img.astype(np.float32)
    noisy = img + np.random.normal(mean, std, img.shape)
    noisy_clipped = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy_clipped

def flip_lr(img):
    return np.fliplr(img)

def flip_ud(img):
    return np.flipud(img)

def change_contrast(img,min_gamma,max_gamma):
    gamma = random.randint(int(min_gamma*100),int(max_gamma*100))/100.0
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    return cv2.LUT(img, lookUpTable)

def random_chop(x,y,min_radio,max_radio):
    radio = random.randint(int(min_radio*100),int(max_radio*100))/100.0
    if radio == 1:
        return x,y

    if radio < 1:
        rows = int(x.shape[0] * radio)
        cols = int(x.shape[1] * radio)
        loc_row = random.randint(0,x.shape[0]-rows)
        col_row = random.randint(0,x.shape[1]-cols)
        x_sub = x[loc_row:loc_row+rows,col_row:col_row+cols]
        y_sub = y[loc_row:loc_row+rows,col_row:col_row+cols]
        x_sub = cv2.resize(x_sub,(x.shape[1],x.shape[0]))
        y_sub = cv2.resize(y_sub,(y.shape[1],y.shape[0]))
        return x_sub,y_sub

    if radio > 1:
        _x = np.zeros(shape=x.shape,dtype=x.dtype)
        _y = np.zeros(shape=y.shape,dtype=y.dtype)

        small_size = (int(x.shape[1]/radio),int(y.shape[0]/radio))
        x_resize = cv2.resize(x,small_size)
        y_resize = cv2.resize(y,small_size)

        rows = int(x.shape[0] / radio)
        cols = int(x.shape[1] / radio)
        loc_row = random.randint(0,x.shape[0]-rows)
        col_row = random.randint(0,x.shape[1]-cols)

        _x[loc_row:loc_row+rows,col_row:col_row+cols] = x_resize
        _y[loc_row:loc_row+rows,col_row:col_row+cols] = y_resize
        return _x,_y

