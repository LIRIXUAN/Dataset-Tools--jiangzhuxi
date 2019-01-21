import os
import numpy as np
import cv2
import shutil
import json
from progressbar import ProgressBar
from preprocess import *

def check_path(path):
    if not os.path.exists(path):
        raise ValueError('Path not exists', path)
    return path

def get_expanded_image(x,y,config):
    args = config['arguments']
    max_std = args['gaussian_noise']['max_standard_deviation']
    min_std = args['gaussian_noise']['min_standard_deviation']

    max_mean = args['gaussian_noise']['max_mean']
    min_mean = args['gaussian_noise']['min_mean']

    max_gamma = args['contrast_change']['max_gamma']
    min_gamma = args['contrast_change']['min_gamma']

    max_radio = args['random_chop']['max_radio']
    min_radio = args['random_chop']['min_radio']
    
    _x = x.copy()
    _y = y.copy()
    if args['random_flip']["left_right"]:
        if random.randint(0,1) == 0:
            _x = flip_lr(_x)
            _y = flip_lr(_y)

    if args['random_flip']["up_down"]:
        if random.randint(0,1) == 0:
            _x = flip_ud(_x)
            _y = flip_ud(_y)

    _x = add_noise(_x,min_std, max_std, min_mean, max_mean)
    _x = change_contrast(_x,min_gamma,max_gamma)
    _x,_y = random_chop(_x,_y,min_radio,max_radio)

    return _x,_y

if __name__ == '__main__':
    json_str = open(check_path('./config.json'), 'r').read()
    exp_config = json.loads(json_str)

    src = exp_config['job']['src_path']
    dst = exp_config['job']['dst_path']

    src_config_path = check_path(os.path.join(src, 'config.json'))
    json_str = open(src_config_path, 'r').read()
    config = json.loads(json_str)

    if not os.path.exists(dst):
        os.mkdir(dst)
    for path in os.listdir(src):
        if os.path.isdir(os.path.join(src,path)):
            if not os.path.exists(os.path.join(dst,path)):
                os.mkdir(os.path.join(dst,path))

    expand_radio = exp_config['arguments']['expand_radio']

    output_index = 0
    progress = ProgressBar()
    start = config['train']['start']
    end = config['train']['end']+1
    for index in progress(range(start,end)):
        x_path = check_path(os.path.join(src, config['train']['x_fmt']) % (index))
        y_path = check_path(os.path.join(src, config['train']['y_fmt']) % (index))
        x,y = cv2.imread(x_path),cv2.imread(y_path)

        for i in range(0,expand_radio):
            _x,_y = get_expanded_image(x,y,exp_config)
            cv2.imwrite(os.path.join(dst, config['train']['x_fmt']) % (output_index), _x)
            cv2.imwrite(os.path.join(dst, config['train']['y_fmt']) % (output_index), _y)
            output_index += 1

            if not exp_config['job']['quiet']:
                expand = cv2.addWeighted(_x,0.5,_y,0.5,20)
                raw = cv2.addWeighted(x,0.5,y,0.5,20)
                cv2.imshow('raw',raw)
                cv2.imshow('expand',expand)
                cv2.waitKey(1)

