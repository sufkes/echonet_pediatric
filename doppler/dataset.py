#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as transforms
from PIL import Image


def selfNormalize(tensor):    
    return (tensor - tensor.mean())/tensor.std()

## Define a dataset class for the doppler images. Needs __init__, __getitem__, and __len__.
class DopplerDataset(Dataset):
    def __init__(self,
                 split = 'train',
                 target_type = 'AOVTI_px',
                 normalize_mode = 'training_set',
                 target_transform = None,
                 file_list_path = '/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/data/data_from_sickkids/processed/clinical_data/vti/FileList_vti.csv',
                 split_col = 'split_all_random',
                 file_path_col = 'FilePath',
                 downscale_y = 10000 # divide return number of pixels by this number to avoid (overflow errors? nonconverging loss?). In early tests predicting the raw number of pixels, the loss would not converge. Dividing by a large number seemed to fix the problem.
    ):
        
        data_df = pd.read_csv(file_list_path)
        data_df = data_df.loc[data_df[split_col] == split, :] # only the portion of the dataframe with the requested split (i.e. separate dataframes for training and validation sets)
        data_df.reset_index(drop=True, inplace=True)
        self.data_df = data_df
        
        self.split = split
        self.target_type = target_type
        if normalize_mode is None:
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor() # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1) or if the numpy.ndarray has dtype = np.uint8. In the other cases, tensors are returned without scaling.
            ])
        elif normalize_mode == 'training_set':

            # Read the mean and standard deviation of the training set from the spreadsheet.
            self.train_mean = data_df.loc[data_df.index[0], 'train_mean'] / 255 # need value in range [0, 1]
            self.train_std = data_df.loc[data_df.index[0], 'train_std'] / 255 # need value in range [0, 1]
            
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[self.train_mean, self.train_mean, self.train_mean],
                                     std=[self.train_std, self.train_std, self.train_std]
                )
            ])
        elif normalize_mode == 'self':
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                selfNormalize
            ])
        else:
            raise Exception('normalize must be set to "training_set", "self", or None')
        self.target_transform = target_transform
        self.file_list_path = file_list_path
        self.split_col = split_col
        self.file_path_col = file_path_col
        if target_type == 'AOVTI_px':
            self.downscale_y = downscale_y

    def __getitem__(self, index):
        # Get input
        img_path = self.data_df.loc[index, self.file_path_col]
        img = Image.open(img_path) # PIL image
        if self.transform:
            img = self.transform(img)

        # Get output
        if self.target_type == 'peak_array':
            peak_array_path = self.data_df.loc[index, 'peak_array_path']
            target = np.load(peak_array_path)
        else:
            target = self.data_df.loc[index, self.target_type]
            if self.target_transform:
                target = self.target_transform(target)
            if self.target_type == 'AOVTI_px':
                target = target/self.downscale_y # scale down the value because it is ~1e4.
        return img, target

    def __len__(self):
        return len(self.data_df)

if __name__ == '__main__':

    ## Run some tests.

    train_dataset = DopplerDataset(split='train', normalize_mode='training_set')
#    val_dataset = DopplerDataset(split='val')
#    test_dataset = DopplerDataset(split='test')

    out_dir = '../runs/doppler'

    # Get the mean and standard deviation of the training dataset for normalization.
    #pixels = np.array([])
    train_pixels_all = []
    for index in range(len(train_dataset)):
        img, target = train_dataset[index]
        train_pixels_all.extend(np.array(img).flatten())
    mean = np.mean(train_pixels_all)
    std = np.std(train_pixels_all)
    print('Training set mean:', mean)
    print('Training set std:', std)
    print('Training set size:', len(train_dataset))

    if False:
        for index, (img, target) in enumerate(train_dataset):
            # Save an image to see how the transform is working.
            file_path = train_dataset.data_df.loc[index, 'FilePath']
            subject = train_dataset.data_df.loc[index, 'Subject']
            out_name = subject + '.png'
            out_path = os.path.join(out_dir, out_name)
            
            img = transforms.ToPILImage()(img)
            img.save(out_path)
            
            break
