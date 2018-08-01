import numpy as np
import os

import torch

from torchvision import datasets, transforms

from composite_model import Composite_Classifier
from solution_manager import Solution_Manager


class Data_Manager():
    """
    Convenience class to manage datasets, dataloader and data parameters

    Arguments:


    """
    def __init__(self, data_dir, phases, my_transforms, bs):
        self.data_dir = data_dir
        self.phases = phases
        self.transforms = my_transforms

        self.image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                                  self.transforms[x])
                          for x in ['train', 'valid', 'test']}

        self._bs = bs

        self.dataloaders = {x: torch.utils.data.DataLoader(self.image_datasets[x], batch_size=self._bs,
                                                      shuffle=True)
                       for x in ['train', 'valid', 'test']}

        self.dataset_sizes = {x: len(self.image_datasets[x]) for x in ['train', 'valid', 'test']}

        return

    def set_dataloader_bs(self):
            for phase in self.phases:
                self.dataloaders[phase].batch_size = self.bs

    @property
    def bs(self):
        return self._bs

    @bs.setter
    def bs(self, bs):
        self._bs=bs
        self.set_dataloader_bs()

