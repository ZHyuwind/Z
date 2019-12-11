# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import glob
from test import *
import sys
import argparse
from utils_pac import *
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms

import torch.nn as nn
import torch.utils.data
import numpy as np
from main_fast_inference import *

import ntpath
import os
import sys
from tqdm import tqdm
import time
import cv2
import copy

# from pPose_nms import pose_nms, write_json
parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')

parser.add_argument('--resume', default='/home/dev/zy/tracking/SiamMask/experiments/siammask_sharp/SiamMask_VOT.pth', type=str,
                    metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('--config', dest='config', default='/home/dev/zy/tracking/SiamMask/experiments/siammask_sharp/config_vot.json',
                    help='hyper-parameter of SiamMask in json format')
parser.add_argument('--base_path', default='/home/dev/zy/tracking/SiamMask/data/jump', help='datasets')
parser.add_argument('--cpu', action='store_true', help='cpu mode')
# parser.add_argument('--nClasses', default=17, type=int, help='Number of output channel')
# parser.add_argument('--inputResH', default=320, type=int, help='Input image height')
# parser.add_argument('--inputResW', default=256, type=int, help='Input image width')
# parser.add_argument('--outputResH', default=80, type=int, help='Output heatmap height')
# parser.add_argument('--outputResW', default=64, type=int, help='Output heatmap width')
args = parser.parse_args()

if __name__ == '__main__':
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    # Setup Model
    cfg = load_config(args)
    from custom import Custom
    siammask = Custom(anchors=cfg['anchors'])
    if args.resume:
        assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
        siammask = load_pretrain(siammask, args.resume)

    siammask.eval().to(device)

    # Parse Image file
    img_files = sorted(glob.glob(join(args.base_path, '*.jp*')))
    ims = [cv2.imread(imf) for imf in img_files]

    # Select ROI
    cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty("SiamMask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    try:
        init_rect = cv2.selectROI('SiamMask', ims[0], False, False)
        x, y, w, h = init_rect #center,width,height
    except:
        exit()

    # Load pose model
    pose_dataset = Mscoco()
    pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
    pose_model.cuda()
    pose_model.eval()

    toc = 0
    for f, im in enumerate(ims):

        tic = cv2.getTickCount()
        if f == 0:  # init
            target_pos = np.array([x + w / 2, y + h / 2])
            target_sz = np.array([w, h])
            state = siamese_init(im, target_pos, target_sz, siammask, cfg['hp'], device=device)  # init tracker
            pt1 = np.array([x-w/2,y-h/2])
            pt2 = target_pos
            box = np.array([pt1[0],pt1[1],pt2[0],pt2[1]])
            # img = copy.deepcopy(im)
            inps = torch.zeros(1, 3, opt.inputResH, opt.inputResW)  # ????????
            inp = im_to_torch(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            inps, pt1, pt2 = crop_from_dets(inp, box, inps, pt1, pt2)
            inps = inps.unsqueeze(0).cuda()
            hm = pose_model(inps)
            hm = hm.cpu().data
            preds_hm, preds_img, preds_scores = getPrediction(hm, pt1, pt2, opt.inputResH, opt.inputResW, opt.outputResH, opt.outputResW)
            for points in preds_img[0]:
                cv2.circle(im,tuple(np.array(points)),radius=3,color=(0,200,0),thickness=4)
            # display_pose(preds_img,img_files[f])
            # visualization(preds_img)
            print(preds_img)
        elif f > 0:  # tracking
            state = siamese_track(state, im, mask_enable=True, refine_enable=True, device=device)  # track
            location = state['ploygon'].flatten()

            #use location to get box than according to alphapose to obtain keypoints
            box = []
            pt1 = []
            pt2 = []
            xmin = min(location[0], location[2], location[4], location[6])
            xmax = max(location[0], location[2], location[4], location[6])
            ymin = min(location[1], location[3], location[5], location[7])
            ymax = max(location[1], location[3], location[5], location[7])
            mask = state['mask'] > state['p'].seg_thr
            box.append(xmin)
            box.append(xmax)
            box.append(ymin)
            box.append(ymax)
            pt1.append(xmin)
            pt1.append(ymin)
            pt2.append(xmax)
            pt2.append(ymax)
            #according to box to estimate pose
            # img = copy.deepcopy(im)
            inps = torch.zeros(1, 3, opt.inputResH, opt.inputResW)#????????
            inp = im_to_torch(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            inps, pt1, pt2 = crop_from_dets(inp, box, inps, pt1, pt2)
            inps = inps.unsqueeze(0).cuda()
            hm = pose_model(inps)
            hm = hm.cpu().data
            preds_hm, preds_img, preds_scores = getPrediction(hm, pt1, pt2, opt.inputResH, opt.inputResW, opt.outputResH, opt.outputResW)
            for points in preds_img[0]:
                cv2.circle(im,tuple(np.array(points)),radius=3,color=(0,255,0),thickness=4)
            # display_pose(preds_img, img_files[f])
            # visualization(preds_img)
            print(preds_img)
            im[:, :, 2] = (mask > 0) * 100 + (mask == 0) * im[:, :, 2]
            cv2.polylines(im, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
            cv2.imshow('SiamMask', im)
            key = cv2.waitKey(1)
            if key > 0:
                break

        toc += cv2.getTickCount() - tic
    toc /= cv2.getTickFrequency()
    fps = f / toc
    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))

