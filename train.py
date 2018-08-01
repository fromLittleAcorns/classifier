'''
Script to train a neural network to learn to classify flowers.  Command line arguments are:

        Basic usage: python train.py data_directory
        Prints out training loss, validation loss, and validation accuracy as the network trains
        Options:
            Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
            Choose architecture: python train.py data_dir --arch "vgg13"
            Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
            Use GPU for training: python train.py data_dir --gpu

Note - at present the model can cope with multilayer classifiers but the command line arguments will only allow a single
layer to be setup.  This can be modified in future but I wanted to be compatible with the above argument structure

'''


import argparse
import os

import sys

import matplotlib.pyplot as plt
from PIL import Image

import numpy as np
import pandas as pd
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, transforms
from torchvision import models as models

# Import my classes
from composite_model import Composite_Classifier
from solution_manager import  Solution_Manager
from data_manager import Data_Manager
import utility


# code to load model names as used by Pytorch Imagenet example
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(description='Helen Richmond Flower Classifier')

# Define directory for datafiles
parser.add_argument('data_directory', type=str,
                    help='path to dataset', default='flowers')

# Define directory where checkpoint files should be saved or loaded from
parser.add_argument('--save_dir', type=str, default='checkpoint',
                    help='directory to save checkpoint files')

parser.add_argument('--resume', default='', type=str,
                    help="Path to directory to load checkpoint from (default = '') ")

parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18), note - ignored when resume option specified')

# Define hyper parameter learning rate
parser.add_argument('--learning_rate', '--lr',  type=float,
                    help='learning rate to use', default=0.01)

parser.add_argument('-bs', '--batch-size', default=16, type=int,
                    help='batch size (default: 16)')

# Define hyper parameter Epochs
parser.add_argument('--epochs', metavar='EPOCHS', type=int,
                    help='Number of epochs to run', default=20)

# Define whether to use GPU for training
parser.add_argument('--gpu', dest='gpu', action='store_true',
                    help='Use GPU for training')

parser.add_argument('--hidden_units', type=str,
                    help='number of neurons in hidden layer', default='[512,256]')

# Define hyper parameter hidden layer dropout
parser.add_argument('--dropout', type=str,
                    help='dropout to use for hidden layer eg 0.5', default='[0.2, 0.2]')


def main():

    global args
    args = parser.parse_args()

    # Default settings
    default_model = 'resnet18'
    default_bs = 16
    sz = 224

    # Take actions based upon initial arguments

    if args.gpu:
        # Check for GPU and CUDA libraries
        HAS_CUDA = torch.cuda.is_available()
        if not HAS_CUDA:
            sys.exit('No Cuda capable GPU detected')
    else:
        HAS_CUDA=False

    checkpoint_dir = args.save_dir

    # Define hyper-parameters

    # Note - allow dropout to be changed when resuming model

    tmp = args.dropout
    tmp = re.sub("[\[\]]", "", tmp)
    drops = [float(item) for item in tmp.split(',')]

    lr = args.learning_rate

    epochs = args.epochs


    # All arguments imported, will start to setup model depending upon whether restarting from checkpoint or
    # from scratch

    if args.resume:
        if os.path.isdir(args.resume):
            print('Loading checkpoint...')
            sol_mgr, pt_model = utility.load_checkpoint(args.resume, lr, HAS_CUDA)

    else:
        # Define hidden layer details (note - if resuming will continue with values used earlier

        tmp = args.hidden_units
        tmp = re.sub("[\[\]]","",tmp)
        n_hid = [int(item) for item in tmp.split(',')]


        # check data directory exists and assign
        data_dir = args.data_directory
        # Check it exists
        if not os.path.exists(data_dir):
            sys.exit('Data directory does not exist')

        # Create model, datasets etc from scratch
        # create datasets and dataloaders
        phrases = ['train', 'valid', 'test']

        # Define data transforms
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(sz),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'valid': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(sz),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(sz),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        bs = args.batch_size

        data = Data_Manager(data_dir, phrases, data_transforms, bs)

        # Load cat_to_name
        cat_to_name = utility.load_classes('cat_to_name.json')
        num_cat = len(cat_to_name)

        # Load pre-trained model
        if args.arch is not None:
            pt_model = args.arch
        else:
            pt_model = default_model
        model_pt = models.__dict__[pt_model](pretrained=True)
        num_ftrs = model_pt.fc.in_features

        # Create classifier model
        img_cl = Composite_Classifier(model_pt, n_hid, drops, num_cat)
        # Move to CUDA if available
        if HAS_CUDA:
            img_cl.cuda()

        # Define losses and hyper-parameters
        criterion = nn.CrossEntropyLoss()
        # Optimise just the parameters of the classifier layers
        optimizer_ft = optim.SGD(img_cl.cf_layers.parameters(), lr=lr, momentum=0.9)

        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

        # Freeze the pre-trained model layers
        for param in img_cl.model_pt.parameters():
            param.requires_grad = False

        # Create model manager to control training, validation, test and predict with the model and data
        sol_mgr = Solution_Manager(img_cl, criterion, optimizer_ft,
                           exp_lr_scheduler, data, phrases, HAS_CUDA=HAS_CUDA)
        sol_mgr.model.class_to_idx = data.image_datasets['train'].class_to_idx

    # Train model
    sol_mgr.train(epochs=epochs)

    # Evaluate model against test set
    sol_mgr.test_with_dl()

    # Save Checkpoint
    utility.save_checkpoint(args.save_dir, sol_mgr, pt_model, HAS_CUDA)


if __name__ == '__main__':
    main()


