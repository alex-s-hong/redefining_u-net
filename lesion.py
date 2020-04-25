import os
import os.path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import random

class LesionDataset(Dataset):
    """Face Landmarks dataset."""
   
    def __init__(self, root_dir, folder_name = "train", joint_transform = None,img_transform = False,seg_transform=None,verbose = False):
        self.root_dir = root_dir
        self.folder_name = folder_name
        self.joint_transform = joint_transform
        self.img_transform = img_transform
        self.seg_transform = seg_transform
        self.data_type = None
        if(self.folder_name == "train"):
            self.data_type = "Training"
        elif(self.folder_name == "val"):
            self.data_type = "Validation"
        else:
            self.data_type = "Test"

        self.img_folder = "ISIC-2017_" + self.data_type + "_Data"
        self.seg_folder = "ISIC-2017_" + self.data_type + "_GroundTruth"

        img_folder_path = os.path.join(root_dir, self.folder_name,self.img_folder)
        seg_folder_path = os.path.join(root_dir, self.folder_name,self.seg_folder)
        self.verbose = verbose
        self.img_filenames = []
        self.seg_filenames = []



        for filename in os.listdir(img_folder_path):
            if(filename.endswith(".jpg")):
                self.img_filenames.append(os.path.join(img_folder_path, filename))

        for filename in os.listdir(seg_folder_path):
            if(filename.endswith(".png")):
                self.seg_filenames.append(os.path.join(seg_folder_path, filename))
                
        self.img_filenames.sort()
        self.seg_filenames.sort()



    def __len__(self):
        return len(self.img_filenames)
    
    def transform(self, image, mask):
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            return image,mask

    def __getitem__(self, idx):
        img_path = self.img_filenames[idx]
        seg_path = self.seg_filenames[idx]
        
        image = Image.open(img_path).convert('RGB')
        target = Image.open(seg_path)
        
        #pillow uses wxh, pytorch uses hxw
        image = image.resize((1024,512))
        target = target.resize((1024,512))
        
        
        
        if(self.joint_transform):
            image,target = self.transform(image,target)
        if(self.img_transform):
            image= self.img_transform(image)
        if(self.seg_transform):
            target = self.seg_transform(target)
        if(self.verbose):
            return image,target,img_path,seg_path
        return image,target
        
        



