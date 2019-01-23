import os
import cv2
import numpy as np
from preprocess import *

if __name__ == '__main__':
    x = cv2.imread('./demo_images/kitti_0.jpg')
    y = cv2.imread('./demo_images/kitti_mask_0.png')
    while True:
        _x = x.copy()
        _y = y.copy()
        if random.randint(0,1) == 0:
            _x = flip_lr(_x)
            _y = flip_lr(_y)
        _x = add_noise(_x,0, 30, -5, 5)
        _x = change_contrast(_x,0.7,1.3)
        _x,_y = random_chop(_x,_y,0.8,1.2)

        expand = cv2.addWeighted(_x,0.5,_y,0.5,20)
        raw = cv2.addWeighted(x,0.5,y,0.5,20)
        cv2.imshow('raw',raw)
        cv2.imshow('expand',expand)
        cv2.waitKey(500)
        cv2.imwrite('./demo_images/kittichange.jpg',expand)

