# -*- coding: utf-8 -*-
"""
Bone Age Assessment Network (BoNet) PyTorch implementation.
"""
import os
import pdb
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter
import torchvision.transforms.functional as TF

# ========================================================================
# Auxiliary functions
# ========================================================================
# ========================================================================
def load_json(route_json):
    with open(route_json, 'r') as f:
        data = json.load(f)
    return data

# ========================================================================
def extract_channels(img_dir, image_name, rois, crop = False, half= True):
    image = Image.open(os.path.join(img_dir, image_name+'.png'))
    # Loading as PIL.Image, converting to numpy while ensuring grayscale
    image = image.convert('L')
    image = np.array(image) 
    # Getting the image ID in the Json file
    imgs_list = rois['images']
    img_id=[im['id'] for im in imgs_list if im['file_name']==image_name+'.png'][0]
    del imgs_list

    # Getting the image annotations, using said ID 
    annotations = rois['annotations']

    for im_ann in annotations:
        if im_ann['image_id'] == img_id:
            im_kpts = im_ann['keypoints']
            im_bbox = (list(map(int, im_ann['bbox'])))
            break
    
    del annotations
    del rois  # Cleaning memory 

    im_kpts = np.array(im_kpts, np.float32)
    im_kpts.shape = (17, 3)
    im_kpts = np.delete(im_kpts, 2, 1)
    im_kpts.reshape(1, -1, 2)
    im_kpts = (im_kpts).astype(int)
    if np.max(im_kpts)==0:
        w,h=image.shape
        im_bbox=[0,0,w,h]
    hmimg = blurr_kpts(image, im_kpts)
    
    if half:
        w,h=image.shape
        im_bbox=[0,0,w//2,h]

    if crop or half:
        image = crop_img(image, im_bbox)
        hmimg = crop_img(hmimg, im_bbox)

    joint_image = np.zeros((2,image.shape[0], image.shape[1]))
    joint_image[0,:, :] = image
    joint_image[1,:, :] = hmimg


    return joint_image

# ========================================================================
def crop_img(image, bbox):
    cropped = image[bbox[1]:bbox[1]+bbox[3],
                          bbox[0]:bbox[0]+bbox[2]]
    
    return cropped

# ========================================================================
def crop_original(image_name, img, rois,half):

    imgs_list = rois['images']
    img_id=[im['id'] for im in imgs_list if im['file_name']== image_name+'.png'][0]
    del imgs_list

    # Getting the image annotations, using said ID 
    annotations = rois['annotations']

    for im_ann in annotations:
        if im_ann['image_id'] == img_id:
            im_bbox = (list(map(int, im_ann['bbox'])))
            break

    del annotations
    del rois  # Cleaning memory 
    if np.max(im_bbox) ==0:
        w,h=img.shape
        im_bbox=[0,0,w,h]
    if half:
        w,h=img.shape
        im_bbox=[0,0,w//2,h]

    img = crop_img(img, im_bbox)

    return img


# ========================================================================
def blurr_kpts(image, kpts):
    kpt_blur = np.zeros((3, 3, 1))
    blurring = np.zeros((image.shape[0], image.shape[1]))
    if np.max(kpts) !=0:
        range_grid = np.arange(-1, 2)
        sigma = 25 # Asuming we want to have blurs over a 50x50 window
        [x_grid, y_grid] = np.meshgrid(range_grid, range_grid)
        kpt_blur = x_grid**2+y_grid**2
        kpt_blur = kpt_blur/(2*sigma**2)
        kpt_blur = np.exp(-kpt_blur)*255
        # Placing the gaussian over each point
        for i in range(len(kpts)):
            try:
                min_row = int(kpts[i, 1]-1)
                max_row = int(min_row+kpt_blur.shape[0])
                min_col = int(kpts[i, 0]-1)
                max_col = int(min_col+kpt_blur.shape[1])
                blurring[min_row:max_row, min_col:max_col] = kpt_blur
            except IndexError:  # If the window/kpt is out of bounds for the image
                continue

        # Blurring the kpts
        blurring = gaussian_filter(blurring, sigma)
    return blurring


# ========================================================================
# Classes - Dataloaders
# ========================================================================
class BoneageDataset(Dataset):
    """Bone Age Assessment dataset."""

    def __init__(self, img_dir, ann_file, json_file=None, img_transform=None, crop=False, dataset='RSNA'):
        """
        Args:
            img_dir (string): Directory with all the images.
            ann_file (string): Path to the csv file with annotations.
            img_transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img_dir = img_dir
        self.annotations = pd.read_csv(ann_file)#,dtype=object)
        self.img_transform = img_transform
        self.crop=crop
        self.kpts = load_json(json_file)
        self.half= dataset=='RHPE' and not crop
        self.dataset=dataset

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = idx.item()
        info = self.annotations.iloc[idx]

        if self.dataset == 'RHPE':
            image_name=str(info[0]).zfill(5)
        else:
            image_name=str(info[0])

        img = np.array(Image.open(os.path.join(self.img_dir, image_name+'.png')).convert('L'))
        bone_age = torch.tensor(info[2], dtype=torch.float)
        gender = torch.tensor(info[1]*1, dtype=torch.float).unsqueeze_(-1)

        if self.crop:
            img=crop_original(image_name,img,self.kpts,half=False)
        if self.half:
            img=crop_original(image_name,img,self.kpts,half=True)

        if self.img_transform:
            out_img = self.img_transform(Image.fromarray(img))
        else:
            out_img = img

        return out_img, bone_age, gender, info[0]

# ========================================================================
class Boneage_HeatmapDataset(Dataset):
    """Bone Age with heatmaps dataset"""
    def __init__(self, img_dir, ann_file, json_file, img_transform = None,
            crop = False,dataset='RSNA'):
        self.annotations = pd.read_csv(ann_file)
        self.crop = crop
        self.img_dir = img_dir
        self.kpts = load_json(json_file)
        self.img_transform = img_transform
        self.half= dataset=='RHPE' and not crop
        self.dataset=dataset

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = idx.item()
        info = self.annotations.iloc[idx]

        if self.dataset == 'RHPE':
            image_name=str(info[0]).zfill(5)
        else:
            image_name=str(info[0])

        img = extract_channels(self.img_dir, image_name, self.kpts,
                self.crop,self.half)
        bone_age = torch.tensor(info[2], dtype = torch.float)
        gender = torch.tensor(info[1]*1, dtype = torch.float).unsqueeze(-1)

        if self.img_transform:

            x_ray = self.img_transform(Image.fromarray(img[0, :, :]))
            h_map = self.img_transform(Image.fromarray(img[1, :, :]))
            out_im = torch.zeros(2, x_ray.shape[1], x_ray.shape[2])
            out_im[0, :, :] = x_ray
            out_im[1, :, :] = h_map
        else:
            out_im = img

        return out_im, bone_age, gender, info[0]

    def __len__(self):
        return len(self.annotations)


