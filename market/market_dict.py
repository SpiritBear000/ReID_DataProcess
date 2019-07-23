import os
import pandas as pd
import re
from glob import glob
import json
import sys
import random
import csv

#############
# arguments #
#############

osp = os.path
save_path = './image_dict'
if not os.path.isdir(save_path):
    os.mkdir(save_path)

market_train_path = './data/market1501/bounding_box_train'
filename = 'image_dict.json'

#####################
# Create Image_dict #
#####################

if os.path.isfile(osp.join(save_path, filename)):

    with open(osp.join(save_path, filename)) as f_obj:
        image_dict = json.load(f_obj)
    # print(image_dict.keys())
    # print(image_dict['2'].items())
    # print(image_dict[str(485)][str(1)])

else:
    all_pids = {}
    all_camid = {}
    pattern = re.compile(r'([-\d]+)_c(\d)')
    fpaths = sorted(glob(osp.join(market_train_path, '*.jpg')))
    for fpath in fpaths:
        fname = osp.basename(fpath)
        pid, cam = map(int, pattern.search(fname).groups())
        if pid == -1: continue
        if pid not in all_pids:
            all_pids[pid] = len(all_pids)
        if cam not in all_camid:
            all_camid[cam] = len(all_camid)

    from collections import OrderedDict
    image_dict = OrderedDict()
    index_list = list(range(len(fpaths))) # for accelerating retrieval speed
    for pid in list(all_pids.keys()):
        image_dict[pid] = OrderedDict()
        for cam in range(1, len(all_camid)+1):
            image_dict[pid][cam] = []
            index_list_ = index_list.copy() # just for accelerating retrieval speed
            for index in index_list:
                fbname = osp.basename(fpaths[index])
                id, cid = map(int, pattern.search(fbname).groups())
                if pid == id and cam == cid:
                    image_dict[pid][cam].append(fbname)
                    index_list_.remove(index)
            index_list = index_list_.copy() # just for accelerating retrieval speed

    with open(osp.join(save_path, filename), 'w') as f_obj:
        json.dump(image_dict, f_obj)

# -------------- image_dict shape -----------------
print()
# image_dict = {pid1:{ cam1:['im1_name.jpg',
#                            'im2_name.jpg',
#                            ...],
#                      cam2:['im1_name.jpg',
#                            'im2_name.jpg',
#                            ...],
#                      ... },
#               pid2:{ cam1:['im1_name.jpg',
#                             'im2_name.jpg',
#                             ...],
#                      cam2:['im1_name.jpg',
#                            'im2_name.jpg',
#                              ...],
#                       ...},
#               ...
#               }
#----------------------------------------------------