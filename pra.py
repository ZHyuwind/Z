import os.path as osp
import numpy as np
import pickle
# import json
# path = "/home/dev/zy/TF-SimpleHumanPose/data/COCO/annotations/person_keypoints_val2017.json"
# with open(path,'r') as f:
#     file = json.load(f)
# print(file.keys())
# print(type(file))
# print("#####################################################################")
# print(file['images'][0].keys())
# print(type(file['images'][0]))
# print(len(file))
# print(file[0])

# print("*********************************************************************")
# print(file['annotations'][0].keys())
# print(file['annotations'][0])
# print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
# print(file['categories'][0].keys())
# print(file['categories'][0])
# print(len(file['images']))
# print(len(file['annotations']))
# print(len(file['categories']))
# curPath = osp.dirname(osp.abspath(__file__))
# # path = rootPath.split('/',4)
# root_dir = osp.join(curPath,'..')
# data_dir = osp.join(root_dir, 'data')
#
# print(curPath)
# print(root_dir)
# print(data_dir)
path = '/home/dev/zy/InstaMoveVision/mask_data/tmp/21979d66-1063-44d0-afb0-5e56a5adf699.pose'
with open(path,'rb') as pickle_file:
    file = pickle.load(pickle_file)
    # pose = poses = list(map(lambda p: CocoPose.from_predicts(p['index'], p['predicts'], p['scores'], p['target_pos']), pickles))


    print(type(file))

    print(file[0])
