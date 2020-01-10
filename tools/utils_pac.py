import os
import torch
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
from tqdm import tqdm
import cv2
import json
import numpy as np
import sys
import time
import argparse
from opt import opt
import matplotlib.pyplot as plt
from PIL import Image


class Mscoco(data.Dataset):
    def __init__(self, train=True, sigma=1,
                 scale_factor=(0.2, 0.3), rot_factor=40, label_type='Gaussian'):
        self.img_folder = '../data/coco/images'    # root image folders
        self.is_train = train           # training set or test set
        self.inputResH = opt.inputResH
        self.inputResW = opt.inputResW
        self.outputResH = opt.outputResH
        self.outputResW = opt.outputResW
        self.sigma = sigma
        self.scale_factor = scale_factor
        self.rot_factor = rot_factor
        self.label_type = label_type

        self.nJoints_coco = 17
        self.nJoints_mpii = 16
        self.nJoints = 33

        self.accIdxs = (1, 2, 3, 4, 5, 6, 7, 8,
                        9, 10, 11, 12, 13, 14, 15, 16, 17)
        self.flipRef = ((2, 3), (4, 5), (6, 7),
                        (8, 9), (10, 11), (12, 13),
                        (14, 15), (16, 17))

    def __getitem__(self, index):
        pass

    def __len__(self):
        pass

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def torch_to_im(img):
    img = to_numpy(img)
    img = np.transpose(img, (1, 2, 0))  # C*H*W
    return img

def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor

def cropBox(img, ul, br, resH, resW):
    ul = ul.int()
    br = (br - 1).int()
    # br = br.int()
    lenH = max((br[1] - ul[1]).item(), (br[0] - ul[0]).item() * resH / resW)
    lenW = lenH * resW / resH
    if img.dim() == 2:
        img = img[np.newaxis, :]

    box_shape = [(br[1] - ul[1]).item(), (br[0] - ul[0]).item()]
    pad_size = [(lenH - box_shape[0]) // 2, (lenW - box_shape[1]) // 2]
    # Padding Zeros
    if ul[1] > 0:
        img[:, :ul[1], :] = 0
    if ul[0] > 0:
        img[:, :, :ul[0]] = 0
    if br[1] < img.shape[1] - 1:
        img[:, br[1] + 1:, :] = 0
    if br[0] < img.shape[2] - 1:
        img[:, :, br[0] + 1:] = 0

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)

    src[0, :] = np.array(
        [ul[0] - pad_size[1], ul[1] - pad_size[0]], np.float32)
    src[1, :] = np.array(
        [br[0] + pad_size[1], br[1] + pad_size[0]], np.float32)
    dst[0, :] = 0
    dst[1, :] = np.array([resW - 1, resH - 1], np.float32)

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    dst_img = cv2.warpAffine(torch_to_im(img), trans,
                             (resW, resH), flags=cv2.INTER_LINEAR)

    return im_to_torch(torch.Tensor(dst_img))

def im_to_torch(img):
    img=np.array(img)
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()
    if img.max() > 1:
        img /= 255
    return img

def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray

def crop_from_dets(img, boxes, inps):
    '''
    Crop human from origin image according to Dectecion Results
    '''

    imght = img.size(1)
    imgwidth = img.size(2)
    tmp_img = img
    tmp_img[0].add_(-0.406)
    tmp_img[1].add_(-0.457)
    tmp_img[2].add_(-0.480)# 这3个参数怎么来的？？

    upLeft = torch.Tensor((float(boxes[0]), float(boxes[1])))
    bottomRight = torch.Tensor((float(boxes[2]), float(boxes[3])))

    ht = bottomRight[1] - upLeft[1]
    width = bottomRight[0] - upLeft[0]

    scaleRate = 0.3

    # upLeft[0] = max(0, upLeft[0] - width * scaleRate / 2)
    # upLeft[1] = max(0, upLeft[1] - ht * scaleRate / 2)
    # bottomRight[0] = max(
    #         min(imgwidth - 1, bottomRight[0] + width * scaleRate / 2), upLeft[0] + 5)
    # bottomRight[1] = max(
    #         min(imght - 1, bottomRight[1] + ht * scaleRate / 2), upLeft[1] + 5)

    try:
        inps = cropBox(tmp_img.clone(), upLeft, bottomRight, opt.inputResH, opt.inputResW)
        # inps = cropBox(tmp_img.clone(), upLeft, bottomRight, int(boxes[3]-boxes[1]), int(boxes[2]-boxes[0]))
    except IndexError:
        print(tmp_img.shape)
        print(upLeft)
        print(bottomRight)
        print('===')
    pt1 = upLeft
    pt2 = bottomRight

    return inps, pt1, pt2

def getPrediction(hms, pt1, pt2, inpH, inpW, resH, resW):
    '''
    Get keypoint location from heatmaps
    '''

    assert hms.dim() == 4, 'Score maps should be 4-dim'
    maxval, idx = torch.max(hms.view(hms.size(0), hms.size(1), -1), 2)

    maxval = maxval.view(hms.size(0), hms.size(1), 1)
    idx = idx.view(hms.size(0), hms.size(1), 1) + 1

    preds = idx.repeat(1, 1, 2).float()

    preds[:, :, 0] = (preds[:, :, 0] - 1) % hms.size(3)
    preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / hms.size(3))

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    preds *= pred_mask

    # Very simple post-processing step to improve performance at tight PCK thresholds
    for i in range(preds.size(0)):
        for j in range(preds.size(1)):
            hm = hms[i][j]
            pX, pY = int(round(float(preds[i][j][0]))), int(round(float(preds[i][j][1])))
            if 0 < pX < opt.outputResW - 1 and 0 < pY < opt.outputResH - 1:
            # if 0 < pX < round(box[2]-box[0]) - 1 and 0 < pY < round(box[3]-box[1]) - 1:
                diff = torch.Tensor((hm[pY][pX + 1] - hm[pY][pX - 1], hm[pY + 1][pX] - hm[pY - 1][pX]))
                preds[i][j] += diff.sign() * 0.25
    preds += 0.2

    preds_tf = torch.zeros(preds.size())

    preds_tf = transformBoxInvert_batch(preds, pt1, pt2, inpH, inpW,resH, resW)

    return preds, preds_tf, maxval

def flip(x):
    assert (x.dim() == 3 or x.dim() == 4)
    dim = x.dim() - 1
    if '0.4.1' in torch.__version__ or '1.0' in torch.__version__:
        return x.flip(dims=(dim,))
    else:
        is_cuda = False
        if x.is_cuda:
            is_cuda = True
            x = x.cpu()
        x = x.numpy().copy()
        if x.ndim == 3:
            x = np.transpose(np.fliplr(np.transpose(x, (0, 2, 1))), (0, 2, 1))
        elif x.ndim == 4:
            for i in range(x.shape[0]):
                x[i] = np.transpose(
                    np.fliplr(np.transpose(x[i], (0, 2, 1))), (0, 2, 1))
        # x = x.swapaxes(dim, 0)
        # x = x[::-1, ...]
        # x = x.swapaxes(0, dim)

        x = torch.from_numpy(x.copy())
        if is_cuda:
            x = x.cuda()
        return x

def shuffleLR(x, dataset):
    flipRef = dataset.flipRef
    assert (x.dim() == 3 or x.dim() == 4)
    for pair in flipRef:
        dim0, dim1 = pair
        dim0 -= 1
        dim1 -= 1
        if x.dim() == 4:
            tmp = x[:, dim1].clone()
            x[:, dim1] = x[:, dim0].clone()
            x[:, dim0] = tmp.clone()
            #x[:, dim0], x[:, dim1] = deepcopy((x[:, dim1], x[:, dim0]))
        else:
            tmp = x[dim1].clone()
            x[dim1] = x[dim0].clone()
            x[dim0] = tmp.clone()
            #x[dim0], x[dim1] = deepcopy((x[dim1], x[dim0]))
    return x

def transformBoxInvert_batch(pt, ul, br, inpH, inpW, resH, resW):
    '''
    pt:     [n, 17, 2]
    ul:     [n, 2]
    br:     [n, 2]
    '''
    center = (br - 1 - ul) / 2
    # center = torch.tensor([(box[2] -box[0]) / 2, (box[3] - box[1])/2])


    size = br - ul
    # size = torch.tensor([box[2] -box[0], box[3] - box[1]])
    size[0] *= (inpH / inpW)
    # size[0]*=((box[3]-box[1])/(box[2]-box[0]))
    lenH = torch.max(size)# [n,]
    lenW = lenH * (inpW / inpH)
    # lenW = lenH * ((box[2]-box[0]) / (box[3]-box[1]))
    # resH = int(round((box[3]-box[1])/4))
    _pt = (pt * lenH[np.newaxis,np.newaxis]) / resH
    # _pt = (pt * lenH[np.newaxis,np.newaxis]) / int(round(box[3]-box[1]))
    _pt[:, :, 0] = _pt[:, :, 0] - ((lenW[np.newaxis].repeat(1, 17) - 1) /
                                   2 - center[0].unsqueeze(-1).repeat(1, 17)).clamp(min=0)
    _pt[:, :, 1] = _pt[:, :, 1] - ((lenH[np.newaxis].repeat(1, 17) - 1) /
                                   2 - center[1].unsqueeze(-1).repeat(1, 17)).clamp(min=0)

    new_point = torch.zeros(pt.size())
    new_point[:, :, 0] = _pt[:, :, 0] + ul[0].unsqueeze(-1).repeat(1, 17)
    new_point[:, :, 1] = _pt[:, :, 1] + ul[1].unsqueeze(-1).repeat(1, 17)
    return new_point

def display_pose(pose_list,img_name):
    img = Image.open(img_name)
    width, height = img.size
    pose_list = np.array(pose_list[0])
    alpha_ratio = 5.0
    fig = plt.figure(figsize=(width / 10, height / 10), dpi=10)
    plt.imshow(img)
    coco_part_names = ['Nose','LEye','REye','LEar','REar','LShoulder','RShoulder','LElbow','RElbow','LWrist','RWrist','LHip','RHip','LKnee','RKnee','LAnkle','RAnkle']
    colors = ['r', 'r', 'r', 'r', 'r', 'y', 'y', 'y', 'y', 'y', 'y', 'g', 'g', 'g','g','g','g']
    pairs = [[0,1],[0,2],[1,3],[2,4],[5,6],[5,7],[7,9],[6,8],[8,10],[11,12],[11,13],[13,15],[12,14],[14,16],[6,12],[5,11]]
    for idx_c, color in enumerate(colors):
        plt.plot(np.clip(pose_list[idx_c,0],0,width), np.clip(pose_list[idx_c,1],0,height), marker='o', color=color)
    for idx in range(len(pairs)):
        plt.plot(np.clip(pose_list[pairs[idx],0],0,width),np.clip(pose_list[pairs[idx],1],0,height),'r-', color='purple',linewidth=10,alpha=5)

    plt.axis('off')
    ax = plt.gca()
    ax.set_xlim([0, width])
    ax.set_ylim([height, 0])
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.show()
    plt.close()
