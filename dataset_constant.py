# -*- coding: utf-8 -*-
# @Time    : 2020/12/10 11:12
# @Author  : Fusen Wang
# @Email   : 201924131014@cqu.edu.cn
# @File    : dataset_constant.py
# @Software: PyCharm


from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import torch
import cv2
from config import *
import random
import torchvision.transforms as transforms
import re

def divideByfactor(img,gt_dmap, factor=16):
    shape1,shape2 = gt_dmap.shape[0], gt_dmap.shape[1]
    crop1, crop2 = 0, 0
    if (shape1 % factor != 0):
        shape1 = int(shape1 // factor * factor)
        crop1 = random.randint(0, shape1 % factor)
    if (shape2 % factor != 0):
        shape2 = int(shape2 // factor * factor)
        crop2 = random.randint(0, shape2 % factor)
    img_ = img[crop1:shape1 + crop1, crop2:shape2 + crop2,:].copy()
    gt_dmap_ = gt_dmap[crop1:shape1 + crop1, crop2:shape2 + crop2].copy()
    return img_, gt_dmap_

def random_crop(img,gt_dmap):
    h, w = gt_dmap.shape[0], gt_dmap.shape[1]
    lt_corner = [(0, 0), (min(h // 2, h - 128), 0), (0, min(w // 2, w - 128)),
                 (min(h // 2, h - 128), min(w // 2, w - 128)),
                 (random.randint(0, min(h // 2, h - 128)), random.randint(0, min(w // 2, w - 128))),
                 (random.randint(0, min(h // 2, h - 128)), random.randint(0, min(w // 2, w - 128))),
                 (random.randint(0, min(h // 2, h - 128)), random.randint(0, min(w // 2, w - 128))),
                 (random.randint(0, min(h // 2, h - 128)), random.randint(0, min(w // 2, w - 128))),
                 (random.randint(0, min(h // 2, h - 128)), random.randint(0, min(w // 2, w - 128)))]
    index = random.randint(0, len(lt_corner) - 1)
    y0, y1 = lt_corner[index][0], lt_corner[index][0] + 128
    x0, x1 = lt_corner[index][1], lt_corner[index][1] + 128
    img_ = img[y0:y1, x0:x1,:].copy()
    gt_dmap_ = gt_dmap[y0:y1, x0:x1].copy()
    return img_, gt_dmap_

def random_Horflip(img, gt_dmap,probability=0.5):
    img_, gt_dmap_ = np.copy(img), np.copy(gt_dmap)
    if random.random() <= probability:
        #img_ = img[:,::-1,:].copy()
        img_ = img[:, ::-1].copy()
        gt_dmap_ = gt_dmap[:,::-1].copy()
    return img_, gt_dmap_

def random_verflip(img, gt_dmap,probability=0.5):
    img_, gt_dmap_ = np.copy(img), np.copy(gt_dmap)
    if random.random() <= probability:
        img_ = img[::-1, :].copy()
        gt_dmap_ = gt_dmap[::-1, :].copy()
    return img_, gt_dmap_

def random_2gray(img,gt_dmap,probability=0.3):
    img_, gt_dmap_ = np.copy(img), np.copy(gt_dmap)
    if random.random() <= probability:
        gray = (img_[:,:,0]).copy()*0.114+(img_[:,:,1]).copy()*0.587+(img_[:,:,2]).copy()*0.299
        img_[:,:,0],img_[:,:,1],img_[:,:,2] = gray,gray,gray
    return img_, gt_dmap_




class Dataset(Dataset):
    '''
    crowdDataset
    '''

    def __init__(self, dataset=DATASET,phase="train", gt_downsample=1):
        '''
        img_root: the root path of img.
        gt_dmap_root: the root path of ground-truth density-map.
        gt_downsample: default is 0, denote that the output of deep-model is the same size as input image.
        '''
        self.phase=phase
        self.name = dataset
        self.img_root = os.path.join(HOME,self.name,"%s_data\\images"%(phase))
        self.gt_dmap_root = os.path.join(HOME,self.name,"%s_data\\ground_truth"%(phase))
        self.gt_downsample = gt_downsample

        self.img_names = [filename for filename in os.listdir(self.img_root) \
                          if os.path.isfile(os.path.join(self.img_root, filename))]
        random.shuffle(self.img_names)
        self.n_samples = len(self.img_names)
        # self.img_names.sort(key = lambda x: int(re.findall(r'\d+', x)[0]))
        # print(self.img_names)
        # self.transforms = transforms.Compose([transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
        self.transforms = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean= (0.5,),
                                                                   std=(0.5,))])
        self.gt_transforms = transforms.Compose([transforms.ToTensor(),
                                              transforms.Normalize(mean= (0.5,),
                                                                   std=(0.5,))])



    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        img_name = self.img_names[index]

        img = Image.open(os.path.join(self.img_root, img_name))
        #print(type(img))
        img = np.asarray(img,dtype=np.uint8)

        gt_dmap = Image.open(os.path.join(self.gt_dmap_root, img_name.replace('Ct','t')))
        gt_dmap = np.asarray(gt_dmap,dtype=np.uint8)
        #print(img.shape,gt_dmap.shape)
        #print(os.path.join(self.img_root, img_name), os.path.join(self.gt_dmap_root, img_name.replace('Ct','t')))

        if (self.phase == "train") and RANDOM_CROP:
            img, gt_dmap = random_crop(img,gt_dmap)

        if DIVIDE:
            img, gt_dmap = divideByfactor(img,gt_dmap,DIVIDE)
        if (self.phase == "train") and RANDOM_2GRAY:
            img,gt_dmap = random_2gray(img,gt_dmap,RANDOM_2GRAY)
        if (self.phase == "train") and RANDOM_HFLIP:
            img,gt_dmap = random_Horflip(img,gt_dmap,RANDOM_HFLIP)
        if (self.phase == "train") and RANDOM_VFLIP:
            img,gt_dmap = random_verflip(img,gt_dmap,RANDOM_VFLIP)

        if self.gt_downsample > 1:  # to downsample image and density-map to match deep-model.
            ds_rows = int(gt_dmap.shape[0] // self.gt_downsample)
            ds_cols = int(gt_dmap.shape[1] // self.gt_downsample)
            gt_dmap = cv2.resize(gt_dmap, (ds_cols, ds_rows), interpolation=cv2.INTER_CUBIC)
            gt_dmap = gt_dmap[:, :] * self.gt_downsample * self.gt_downsample

        #处理灰度图
        # img = img[:,:,np.newaxis]
        # gt_dmap = gt_dmap[:,:,np.newaxis]
        #print("1,2",img.shape,gt_dmap.shape)
        img_tensor = self.transforms(img)
        #print("3", img_tensor.size())
        # gt_dmap_tensor = torch.tensor(gt_dmap, dtype=torch.float)
        gt_dmap_tensor = self.gt_transforms(gt_dmap)
        #print("4", gt_dmap_tensor.size())

        return img_tensor, gt_dmap_tensor


if __name__ == "__main__":
    import torch.utils.data.dataloader as Dataloader
    import matplotlib.pyplot as plt

    dataset = Dataset(phase="train", gt_downsample=1)
    dataloader = Dataloader.DataLoader(dataset, batch_size=16, shuffle=False, drop_last=False)
    print(len(dataloader))
    for i,(images,targets) in enumerate(dataloader):
        print(images.size(),targets.size())
        # images = images.squeeze(0).transpose(0,2).transpose(0,1)
        # images = images.numpy()
        # print(images)
        # print("------------------------------------------")
        # targets = targets.numpy()
        # print(targets)
        # images[:,:,0],images[:,:,2] = images[:,:,2],images[:,:,0]

        # cv2.imwrite("sample/image.png",images * 255.0)
        #
        # targets = targets.squeeze(0).squeeze(0)
        # plt.imsave("sample/gt_map.png", targets)
        print("11111111111")
        exit(1)

