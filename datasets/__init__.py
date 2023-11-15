#######################################
############### MRI ###################
from torch.utils.data import Dataset, random_split
import torch.fft as torch_fft
import nibabel as nib
import cv2
from PIL import Image
from tqdm import tqdm
from scipy import ndimage, misc
########################################
import pandas as pd
import pydicom as dicom
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np


def get_dataset(args, config):
    global transforms

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Grayscale(1),
        torchvision.transforms.Resize(
            (config.data.image_size, config.data.image_size), antialias=True)
    ])

    all_dataset = torchvision.datasets.ImageFolder(
        config.data.data_dir, transform=transforms)
    print("Class to index:",all_dataset.class_to_idx)

    # Get the number of samples
    num_samples = len(all_dataset)

    # Decide on the proportion you want for training
    train_proportion = 0.99

    # Calculate the number of samples in the training and test sets
    num_train = int(num_samples * train_proportion)
    num_test = num_samples - num_train

    # Split the dataset
    dataset, test_dataset = random_split(
        all_dataset, [num_train, num_test])

    print(f'Length of {config.data.dataset} training dataset: {len(dataset)}')
    print(
        f'Length of {config.data.dataset} valid dataset: {len(test_dataset)}')

    return dataset, test_dataset


def logit_transform(image, lam=1e-6):
    image = lam + (1 - 2 * lam) * image
    return torch.log(image) - torch.log1p(-image)


def data_transform(config, X):
    if config.data.uniform_dequantization:
        X = X / 256. * 255. + torch.rand_like(X) / 256.
    if config.data.gaussian_dequantization:
        X = X + torch.randn_like(X) * 0.01

    if config.data.rescaled:
        X = 2 * X - 1.
    elif config.data.logit_transform:
        X = logit_transform(X)

    if hasattr(config, 'image_mean'):
        return X - config.image_mean.to(X.device)[None, ...]

    return X


def inverse_data_transform(config, X):
    if hasattr(config, 'image_mean'):
        X = X + config.image_mean.to(X.device)[None, ...]

    if config.data.logit_transform:
        X = torch.sigmoid(X)
    elif config.data.rescaled:
        X = (X + 1.) / 2.

    return torch.clamp(X, 0.0, 1.0)


def Gamma_correction(img):
    gamma = 0.6
    im = ((img/1)**gamma) * 1
    return im
