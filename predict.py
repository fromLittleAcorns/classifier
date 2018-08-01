"""

Script to use a pre-trained neural network to predict the most likely class of images of flowers along with a
confidence measure for the most likely classes

        Basic usage: python predict.py /path/to/image checkpoint
        Options:
            Return top KKK most likely classes: python predict.py input checkpoint --top_k 3
            Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
            Use GPU for inference: python predict.py input checkpoint --gpu

"""

import argparse
import os
import sys
import json

from PIL import Image

import numpy as np

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
from solution_manager import Solution_Manager
from data_manager import Data_Manager
import utility

# code to load model names as used by Pytorch Imagenet example
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Helen Richmond Flower Classifier')

# Define directory for datafiles
parser.add_argument('path_to_image', type=str,
                    help='path to image (including filename')

parser.add_argument('checkpoint', type=str,
                    help="Directory containing checkpoint ")

# Define whether to use GPU for prediction
parser.add_argument('--gpu', dest='gpu', action='store_true',
                    help='Use GPU for training')

# Define how many class predictions to provide
parser.add_argument('--topk', type=int,
                    help='How many class predictions to show')

# Define cat to names file
parser.add_argument('--category_names', type=str,
                    help='File containing catagory to name dictionary')


def process_image(image, model_mgr):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array

        Note - the transformations are define in the data_transforms['predict'] dictionary as defined earlier

    '''

    image = model_mgr.data.transforms['test'](image).float()

    # Convert back to numpy array since this is what is requested
    image = image.numpy()

    return image


def predict(image_path, model_mgr, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    # open the image
    image = Image.open(image_path)

    # apply transforms
    tr_image = process_image(image, model_mgr)

    # convert to tensor
    pt_image = torch.FloatTensor(tr_image)

    # add additional batch dimension at the start to give bn, 3, h, w
    pt_image = torch.unsqueeze(pt_image, 0)

    # Feed image through the model using the predict method of the model manager class
    pred = model_mgr.predict(pt_image)

    # obtain top 5 class IDs and probabilities
    class_probs, class_ids = pred.topk(5)
    class_probs = class_probs.numpy()

    idx_to_class = [k for k, v in model_mgr.model.class_to_idx.items()]
    model_mgr.idx_to_class = idx_to_class

    class_ids = class_ids.numpy()
    class_ids = np.squeeze(class_ids)

    # Convert class numbers to folder labels
    class_labels = np.zeros(len(class_ids))
    for i, cid in enumerate(class_ids):
        class_labels[i] = idx_to_class[cid]

    return class_probs, class_ids, class_labels, torch.squeeze(pt_image)


def main():
    global args
    args = parser.parse_args()

    # Default settings
    default_model = 'resnet18'
    my_topk = 1
    sz = 224

    # Take actions based upon initial arguments

    if args.gpu:
        # Check for GPU and CUDA libraries
        HAS_CUDA = torch.cuda.is_available()
        if not HAS_CUDA:
            sys.exit('No Cuda capable GPU detected')
    else:
        HAS_CUDA = False

    # Check how many classes to predict (if argument will only predict top class
    if args.topk:
        my_topk = args.topk

    # load cat_to_name file if specified
    if args.category_names:
        names_given = True
        cat_to_names_file = args.category_names
        with open(cat_to_names_file, 'r') as f:
            cat_to_name = json.load(f)

    else:
        names_given = False

    checkpoint_dir = args.checkpoint

    # Recreate model to facilitate prediction

    print('Loading checkpoint...')
    lr = 0.01  # not actually needed but used to avoid problems with function call - need to add named parameters to avoid this
    cf_mgr, pt_model = utility.load_checkpoint(checkpoint_dir, lr, HAS_CUDA)

    # Load image file
    filename = args.path_to_image
    if not os.path.isfile(filename):
        sys.exit('path to image is not a valid file')

    # Run prediction and obtain probabilities of classes plus the image file
    class_probs, class_ids, class_labels, image = predict(filename, cf_mgr, my_topk)

    class_labels_tidy = []
    for i, cln in enumerate(class_labels):
        class_labels_tidy.append(str(int(cln)))

    if names_given:
        class_names = []
        for i, cln in enumerate(class_labels):
            class_names.append(cat_to_name[str(int(cln))])

    # Print out results
    print(f'Predictions for Image {filename}')
    print(f'   Top 5 predicted probabilities: {np.around(class_probs,5)}')
    print(f'   Top 5 predicted class category: {class_labels_tidy}')
    if names_given:
        print(f'   Top 5 predicted class_names: {class_names}')
    print('')


if __name__ == '__main__':
    main()