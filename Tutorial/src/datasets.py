import glob as glob
import os

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets
from tqdm import tqdm

# Required constants.
ROOT_DIR = '../input/GTSRB_Final_Training_Images/GTSRB/Final_Training/Images'
VALID_SPLIT = 0.1
RESIZE_TO = 224 # Image size of resize when applying transforms.
BATCH_SIZE = 128
NUM_WORKERS = 4 # Number of parallel processes for data preparation.

# Training transforms.
class TrainTransforms:
    def __init__(self, resize_to):
        self.transforms = A.Compose([
            A.Resize(resize_to, resize_to),
            A.RandomBrightnessContrast(),
            A.RandomFog(),
            A.RandomRain(),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
                ),
            ToTensorV2()
        ])
    
    def __call__(self, img):
        return self.transforms(image=np.array(img))['image']

# Validation transforms.
class ValidTransforms:
    def __init__(self, resize_to):
        self.transforms = A.Compose([
            A.Resize(resize_to, resize_to),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
                ),
            ToTensorV2()
        ])
    
    def __call__(self, img):
        return self.transforms(image=np.array(img))['image']

def get_datasets(validation_split: float = VALID_SPLIT):
    """
    Function to prepare the Datasets.

    Returns the training and validation datasets along 
    with the class names.
    """
    dataset = datasets.ImageFolder(
        ROOT_DIR, 
        transform=(TrainTransforms(RESIZE_TO))
    )
    dataset_test = datasets.ImageFolder(
        ROOT_DIR, 
        transform=(ValidTransforms(RESIZE_TO))
    )
    dataset_size = len(dataset)

    # Calculate the validation dataset size.
    valid_size = int(validation_split*dataset_size)
    # Radomize the data indices.
    indices = torch.randperm(len(dataset)).tolist()
    # Training and validation sets.
    dataset_train = Subset(dataset, indices[:-valid_size])
    dataset_valid = Subset(dataset_test, indices[-valid_size:])

    return dataset_train, dataset_valid, dataset.classes

def get_data_loaders(dataset_train, dataset_valid, batch_size: int = BATCH_SIZE, num_workers: int = NUM_WORKERS):
    """
    Prepares the training and validation data loaders.

    :param dataset_train: The training dataset.
    :param dataset_valid: The validation dataset.

    Returns the training and validation data loaders.
    """
    train_loader = DataLoader(
        dataset_train, batch_size=batch_size, 
        shuffle=True, num_workers=num_workers
    )
    valid_loader = DataLoader(
        dataset_valid, batch_size=batch_size, 
        shuffle=False, num_workers=num_workers
    )
    return train_loader, valid_loader

def get_test_set():
    """
    Prepares the test dataset.

    Returns the test dataset.
    """
    dataset_test = datasets.ImageFolder(
        ROOT_DIR, 
        transform=(ValidTransforms(RESIZE_TO))
    )
    return dataset_test

class TestDataset(Dataset):
    def __init__(self, image_list, gt_df, image_paths, transform=None):
        """
        Custom Dataset for loading test images.

        Args:
            image_list: List of images in memory.
            gt_df: DataFrame containing ground truth labels.
            image_paths: List of paths to the images.
            transform: Transformations to be applied to the images.
        """
        self.images = image_list
        self.gt_df = gt_df
        self.image_paths = image_paths
        self.transform = transform if transform is not None else ValidTransforms(RESIZE_TO)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image from memory
        orig_image, image  = self.images[idx]

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Get label
        image_name = self.image_paths[idx].split(os.path.sep)[-1]
        label = self.gt_df.loc[image_name].ClassId

        return image, label, orig_image

    @classmethod
    def cache_images(cls, image_paths):
        """
        Cache images into RAM.

        Args:
            image_paths: List of paths to the images.
        Returns:
            List of images loaded into memory.
        """
        images = []

        for path in tqdm(image_paths, total=len(image_paths), desc='Caching images'):
            image = cv2.imread(path)
            orig_image = image.copy()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append((orig_image, image))
        return images
    
    @classmethod
    def build(cls, image_path_parent:str, label_csv_path:str, sign_name_path:str, batch_size:int,num_workers:int,):
        """
        Build a Dataset and DataLoader for the test set.

        Args:
            image_path_parent: Path to the directory containing the images.
            label_csv_path: Path to the CSV file containing the ground truth.
            batch_size: Batch size for the DataLoader.
            num_workers: Number of workers for the DataLoader.
        Returns:
            Dataset and DataLoader for the test set.
            Class names.
        """
        sign_names_df = pd.read_csv(sign_name_path)
        class_names = sign_names_df.SignName.tolist()
        gt_df = pd.read_csv(
            label_csv_path, 
            delimiter=';'
        )
        gt_df = gt_df.set_index('Filename', drop=True)
        # Get image paths.
        image_paths = glob.glob(os.path.join(image_path_parent, '*.ppm'))
        # Cache images into memory.
        images = cls.cache_images(image_paths)
        # Create a Dataset.
        dataset = cls(images, gt_df, image_paths)
        # Create a DataLoader.
        loader = DataLoader(
            dataset, batch_size=batch_size, 
            shuffle=False, num_workers=num_workers
        )
        return dataset, loader, class_names
