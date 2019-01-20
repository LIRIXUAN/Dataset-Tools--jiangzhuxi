import os
import numpy as np
import cv2
import shutil
import json
from progressbar import ProgressBar

def add_noise(img,std = 10):
    img = img.copy().astype(np.float32)
    mean = 0
    noisy = img + np.random.normal(mean, std, img.shape)
    noisy_clipped = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy_clipped

def flip(img):
    img = img.copy()
    return [np.fliplr(img),np.flipud(img)]

def save():
    index = 0
    p = ProgressBar()
    for i in p(range(0,180)):
        frame = cv2.imread('./dataset/kitti/train/frame_%d.jpg'%(i))
        mask = cv2.imread('./dataset/kitti/train_mask/mask_%d.png'%(i))

        noisy = [
            frame,add_noise(frame,15),add_noise(frame,30),add_noise(frame,45),
            np.fliplr(frame),np.fliplr(add_noise(frame,15)),np.fliplr(add_noise(frame,30)),np.fliplr(add_noise(frame,45)),
        ]
        mask_n = [
            mask,mask,mask,mask,
            np.fliplr(mask),np.fliplr(mask),np.fliplr(mask),np.fliplr(mask),
        ]

        for t in range(0,len(noisy)):
            cv2.imwrite('./dataset/kitti_ex/train/frame_%d.jpg'%(index),noisy[t])
            cv2.imwrite('./dataset/kitti_ex/train_mask/mask_%d.png'%(index),mask_n[t])
            index += 1

def inspect():
    for i in ProgressBar()(range(0,180*8)):
        frame = cv2.imread('./dataset/kitti_ex/train/frame_%d.jpg'%(i))
        mask = cv2.imread('./dataset/kitti_ex/train_mask/mask_%d.png'%(i))
        weighted = cv2.addWeighted(frame,0.5,mask,0.5,20)
        cv2.imshow('weighted',weighted)
        cv2.waitKey(0)

if __name__ == '__main__':
    save()
    #inspect()