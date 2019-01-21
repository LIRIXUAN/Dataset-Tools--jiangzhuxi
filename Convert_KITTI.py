import os
import numpy as np
import cv2
import shutil
import json
from progressbar import ProgressBar

def cvt():
    train_index = 0
    test_index = 0

    src_base = ''
    p = ProgressBar()
    for index in p(range(0,200)):
        frame_path = './dataset/KITTI_Raw/image_2/%06d_10.png'%(index)
        mask_path = './dataset/KITTI_Raw/semantic_rgb/%06d_10.png'%(index)
        frame = cv2.imread(frame_path)
        mask = cv2.imread(mask_path)

        mask_dst = np.zeros(shape=(mask.shape[0],mask.shape[1],3),dtype=np.uint8)
        road = np.array([128, 64, 128])
        mask_dst[:,:,0] = cv2.inRange(mask, road-1, road+1)
        car = np.array([142, 0, 0])
        mask_dst[:,:,1] = cv2.inRange(mask, car-1, car+1)

        cv2.imshow('frame',frame)
        cv2.imshow('mask',mask)
        cv2.imshow('mask_dst',mask_dst)
        cv2.waitKey(1)

        if index % 10 == 0:
            cv2.imwrite('./dataset/kitti/test/frame_%d.jpg'%(test_index),frame)
            cv2.imwrite('./dataset/kitti/test_mask/mask_%d.png'%(test_index),mask_dst)
            test_index += 1
        else:
            cv2.imwrite('./dataset/kitti/train/frame_%d.jpg'%(train_index),frame)
            cv2.imwrite('./dataset/kitti/train_mask/mask_%d.png'%(train_index),mask_dst)
            train_index += 1

config = {
    "train": {
        "start": 0,
        "end": 179,
        "step": 1,
        "fmt": "frame_%d.jpg",
        "frame_folder": "train/",
        "mask_folder": "train_mask/"
    },
    "test": {
        "start": 0,
        "end": 19,
        "step": 1,
        "fmt": "mask_%d.png",
        "frame_folder": "test/",
        "mask_folder": "test_mask/"
    }
}

json_str = json.dumps(config)
open('./dataset/kitti/config.json','w').write(json_str)
