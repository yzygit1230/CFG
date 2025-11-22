from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from glob import glob
import random
import matplotlib.pyplot as plt

class BreastSegmentation(Dataset):
    def __init__(self,
                 base_dir='../../data/BreastSlice',
                 phase='train',
                 splitid=2,
                 domain=[1,2,3,4,5],
                 transform=None,
                 normal_toTensor = None,
                 selected_idxs = None
                 ):
        self._base_dir = base_dir
        self.image_list = []
        self.phase = phase
        self.domain_name = {1:'1-WHU', 2:'2-BUSI', 3:'3-DatasetB', 4:'4-DatasetC', 5:'5-cycle-4T'}
        self.image_pool = []
        self.label_pool = []
        self.img_name_pool = []
        self.img_domain_code_pool = []

        self.splitid = splitid
        self.domain = domain
        SEED = 1212
        random.seed(SEED)
        excluded_num = 0
        for i in self.domain:
            self._image_dir = os.path.join(self._base_dir, self.domain_name[i], phase,'image/')
            print('{}  ==>  Loading {} data from: {}'.format(i, phase, self._image_dir))

            imagelist = glob(self._image_dir + '*.png')
            imagelist.sort()
            
            if self.splitid == i and selected_idxs is not None:
                total = list(range(len(imagelist)))
                excluded_idxs = [x for x in total if x not in selected_idxs]
                excluded_num += len(excluded_idxs)
            else:
                excluded_idxs = []
            
            for exclude_id in reversed(sorted(excluded_idxs)):
                imagelist.pop(exclude_id)
            for image_path in imagelist:
                self.image_pool.append(image_path)
                gt_path = image_path.replace('image', 'mask')
                self.label_pool.append(gt_path)
                self.img_domain_code_pool.append(i)
                _img_name = image_path.split('/')[-1]
                self.img_name_pool.append(self.domain_name[i]+'/'+_img_name)
        
        self.weak_transform = transform
        self.normal_toTensor = normal_toTensor
        
        
        print('-----Total number of images in {}: {:d}, Excluded: {:d}'.format(phase, len(self.image_pool), excluded_num))

    def __len__(self):
        return len(self.image_pool)
        
    def __getitem__(self, index):
        if self.phase == 'train':
            _img = Image.open(self.image_pool[index])
            _target = Image.open(self.label_pool[index])
            if _img.mode is 'RGB':
                _img = _img.convert('L')
            if _target.mode is 'RGB':
                _target = _target.convert('L')
            anco_sample = {'image': _img, 'label': _target, 'img_name':self.img_name_pool[index], 'dc': self.img_domain_code_pool[index]}
            if self.weak_transform is not None:
                anco_sample = self.weak_transform(anco_sample)

            anco_sample = self.normal_toTensor(anco_sample)
        else:
            _img = Image.open(self.image_pool[index])
            _target = Image.open(self.label_pool[index])
            if _img.mode is 'RGB':
                _img = _img.convert('L')
            if _target.mode is 'RGB':
                _target = _target.convert('L')
            anco_sample = {'image': _img, 'label': _target, 'img_name':self.img_name_pool[index], 'dc': self.img_domain_code_pool[index]}
            anco_sample = self.normal_toTensor(anco_sample)

        return anco_sample

    def __str__(self):
        return 'Breast(phase=' + self.phase+str(self.splitid) + ')'
