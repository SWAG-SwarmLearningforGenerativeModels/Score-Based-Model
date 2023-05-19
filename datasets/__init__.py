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

    if config.data.dataset == 'xray':

        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Grayscale(1),
            torchvision.transforms.Resize(
                (config.data.image_size, config.data.image_size), antialias=True)
        ])

        all_dataset = torchvision.datasets.ImageFolder(
            config.data.data_dir, transform=transforms)

    if config.data.dataset == 'brain':

        transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Grayscale(1),
            torchvision.transforms.Resize(
                (config.data.image_size, config.data.image_size), antialias=True)
        ])

        data1 = torchvision.datasets.ImageFolder(
            '/home/domainHomes/aokolie/Desktop/Augustine Workspace/SWAG PROJECT/Score-Based-Model-SWAG/Testing', transform=transforms)
        data2 = torchvision.datasets.ImageFolder(
            '/home/domainHomes/aokolie/Desktop/Augustine Workspace/SWAG PROJECT/Score-Based-Model-SWAG/Training', transform=transforms)

        all_dataset = torch.utils.data.ConcatDataset([data1, data2])

    elif config.data.dataset == 'abdomen2dCT':
        class RSNA(Dataset):

            def __init__(self):

                # Attributes
                self.image_size = config.data.image_size
                self.channels = config.data.channels
                self.frame = pd.read_csv(
                    "/home/domainHomes/aokolie/Desktop/Augustine Workspace/SWAG PROJECT/datasets/CTs/datasets/RSNA Pulmonary Embolism/valid_train.csv")

            def __len__(self):
                return (self.frame.shape[0])

            def read_dicom_image(self, path):
                img = dicom.dcmread(path)
                return (img.pixel_array)

            def show_slice_window(self, slice):
                # Set the range of brightness values manually
                min_brightness = -1000
                max_brightness = 1000

                # Clip the brightness values to the specified range
                clipped_image_np = np.clip(
                    slice, min_brightness, max_brightness)

                # scale the brightness values to the full range (0-1)
                scaled_image_np = (
                    clipped_image_np - min_brightness) / (max_brightness - min_brightness)

                return scaled_image_np

            def get_hounsfield_units(self, dicom_file):
                # Read the DICOM file
                ds = dicom.dcmread(dicom_file)

                # Get the pixel data
                pixel_data = ds.pixel_array

                # Apply the rescale slope and rescale intercept to obtain Hounsfield Units
                rescale_slope = float(ds.RescaleSlope)
                rescale_intercept = float(ds.RescaleIntercept)
                hounsfield_units = pixel_data * rescale_slope + rescale_intercept

                return hounsfield_units

            def __getitem__(self, idx):
                if torch.is_tensor(idx):
                    idx = idx.tolist()

                # complete image path and read
                img_path = self.frame['image'].iloc[idx]

                image = self.get_hounsfield_units(img_path)
                image = self.show_slice_window(image)
                resize_image = cv2.resize(image, dsize=(
                    self.image_size, self.image_size))
                img_tensor = torch.from_numpy(resize_image[None, ...]).float()

                return img_tensor, torch.tensor(1)

        all_dataset = RSNA()

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
