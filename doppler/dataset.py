#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as transforms
#from torchvision.io import read_image # doesn't work; wrong version or something.
#from matplotlib import image
from PIL import Image


def scale_y_fn(mean=10497.52677, std=3163.704147):
    # mean (train) = 10497.52677
    # sd (train)   = 3163.704147
    def scale_y(y):
        return (y-mean)/std
    return scale_y



## Define a dataset class for the doppler images. Needs __init__, __getitem__, and __len__.
class DopplerDataset(Dataset):
    def __init__(self,
                 split = 'train',
                 target_type = 'AOVTI_px',
                 transform = transforms.Compose([
                     transforms.Grayscale(num_output_channels=3),
                     transforms.ToTensor(),
                     transforms.Normalize(mean=[0.07710377368714151,0.07710377368714151,0.07710377368714151],
                                          std=[0.16099024828530692,0.16099024828530692,0.16099024828530692]
                     )
                     # Training set mean: 0.07710377368714151
                     # Training set std: 0.16099024828530692
                 ]),
                 #target_transform = scale_y_fn(),
                 target_transform = None,
                 file_list_path = '/hpf/largeprojects/ccmbio/sufkes/echonet_pediatric/data/data_from_sickkids/processed/clinical_data/vti/FileList_vti.csv',
                 split_col = 'split_all_random',
                 file_path_col = 'FilePath'
    ):
        self.split = split
        self.target_type = target_type
        self.transform = transform
        self.target_transform = target_transform
        self.file_list_path = file_list_path
        self.split_col = split_col
        self.file_path_col = file_path_col
        self.downscale_y = 10000

        data_df = pd.read_csv(file_list_path)
        data_df = data_df.loc[data_df[split_col] == split, :] # only the portion of the dataframe with the requested split (i.e. separate dataframes for training and validation sets)
        data_df.reset_index(drop=True, inplace=True)
        self.data_df = data_df

    def __getitem__(self, index):
        img_path = self.data_df.loc[index, self.file_path_col]
        #img = read_image(img_path) # torchvision.io (not in current version?)
        #img = image.imread(img_path) # matplotlib (works but has different axis mapping than pillow?)
        img = Image.open(img_path) # PIL image
        target = self.data_df.loc[index, self.target_type]
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
            #target -= 10497.52677
            #target /= 3163.704147
            # mean (train) = 10497.52677
            # sd (train)   = 3163.704147
        target = target/self.downscale_y
        return img, target

    def __len__(self):
        return len(self.data_df)

if __name__ == '__main__':

    ## Run some tests.

    train_dataset = DopplerDataset(split='train')
#    val_dataset = DopplerDataset(split='val')
#    test_dataset = DopplerDataset(split='test')

    out_dir = '../runs/doppler'

    # Get the mean and standard deviation of the training dataset for normalization.
    pixels = np.array([])
    for index in range(len(train_dataset)):
        img, target = train_dataset[index]
        pixels = np.append(pixels, np.array(img[0, :, :]))
    print(pixels.shape)
    mean = np.mean(pixels)
    std = np.std(pixels)
    print('Training set mean:', mean)
    print('Training set std:', std)


    for index, (img, target) in enumerate(train_dataset):
        print(index)
        print(img.shape)
        print(type(img))
        print(target)

        # Save an image to see how the transform is working.
        file_path = train_dataset.data_df.loc[index, 'FilePath']
        subject = train_dataset.data_df.loc[index, 'Subject']
        out_name = subject + '.png'
        out_path = os.path.join(out_dir, out_name)

        print(subject)
        print(file_path)
        print(img)
        print(np.mean(np.array(img)))
        print(np.std(np.array(img)))
        img = transforms.ToPILImage()(img)
        img.save(out_path)

        break
