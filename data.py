import torch
import torchvision.datasets
from torch.utils.data import Dataset
from typing import List
from PIL import Image


class SplitMNIST(Dataset):
    def __init__(self, mnist_dataset: Dataset, classes: List[int], transform=None, target_transform=None):
        assert len(classes) > 0

        # Find the indices of examples with targets which are one of those given in classes
        split_dataset_idxs = mnist_dataset.targets == classes[0] 
        for target_class in classes[1:]:
            split_dataset_idxs = split_dataset_idxs | (mnist_dataset.targets == target_class) 

        self.data = mnist_dataset.data[split_dataset_idxs]
        self.targets = mnist_dataset.targets[split_dataset_idxs]

        self.transform = transform
        self.target_transform = target_transform
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)