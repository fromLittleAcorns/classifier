import os
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models as models
from data_manager import Data_Manager
from composite_model import Composite_Classifier
from solution_manager import Solution_Manager

def load_checkpoint(dir_name, lr, HAS_CUDA):
    # load model parameters
    
    if HAS_CUDA:
        device=torch.cuda.current_device()
        print(f'Cuda device: {device}')
        checkpoint = torch.load(dir_name + '/' + 'checkpoint.pth.tar', map_location = lambda storage, loc : storage.cuda(device))
        print('Loaded CUDA version')
    else:
        checkpoint = torch.load(dir_name + '/' + 'checkpoint.pth.tar', map_location = 'cpu')

    # load pretrained model
    pt_model = checkpoint['pt_model']
    model_pt = models.__dict__[pt_model](pretrained=True)

    # Recreate model
    img_cl = Composite_Classifier(model_pt, checkpoint['n_hid'], checkpoint['drops'], checkpoint['num_cat'])

    # load model state disctionary
    img_cl.load_state_dict(checkpoint['model'])

    # Recreate optimiser
    optimizer_ft = optim.SGD(img_cl.cf_layers.parameters(), lr=0.001, momentum=0.9)
    optimizer_ft.load_state_dict(checkpoint['optimizer'])
    old_lr = optimizer_ft.param_groups[0]['lr']
    last_epoch_trained_upon = checkpoint['epochs']
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    
    # Now move optimise to GPU if necessary
    if HAS_CUDA:
        for state in optimizer_ft.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    if old_lr != lr:
        optimizer_ft.param_groups[0]['lr'] = lr
    else:
        # if lr has not been updated put scheduler back to where it was
        exp_lr_scheduler.last_epoch = last_epoch_trained_upon

    # Recreate data object
    data = Data_Manager(checkpoint['data_dir'], checkpoint['phases'], checkpoint['data_tfms'], checkpoint['bs'])

    # Recreate model manager class instance
    phases = checkpoint['phases']
    model_mgr = Solution_Manager(img_cl, checkpoint['loss_function'], optimizer_ft, exp_lr_scheduler, data, phases, HAS_CUDA)

    # restore model manager state variables
    model_mgr.epochs = checkpoint['epochs']
    model_mgr.loss_function = checkpoint['loss_function']
    model_mgr.best_accuracy = checkpoint['best_accuracy']
    model_mgr.best_corrects = checkpoint['best_corrects']
    model_mgr.best_loss = checkpoint['best_loss']
    model_mgr.model.class_to_idx = checkpoint['class_to_idx']

    if HAS_CUDA:
        model_mgr.model.cuda()

    # Freeze the pre-trained model layers
    for param in img_cl.model_pt.parameters():
        param.requires_grad = False

    print('Checkpoint loaded')
    return model_mgr, pt_model


def save_checkpoint(chk_dir_name, cf_mgr, pt_model, HAS_CUDA):

    if not os.path.exists(chk_dir_name):
        os.mkdir("./" + chk_dir_name)

    # Save model parameters to facilitate model re-creating model structure

    #if HAS_CUDA:
    #    cf_mgr.model.cpu()

    state = {'pt_model': pt_model,
             'data_dir': cf_mgr.data.data_dir,
             'data_tfms': cf_mgr.data.transforms,
             'phases': cf_mgr.data.phases,
             'bs': cf_mgr.data.bs,
             'n_hid': cf_mgr.model.n_hid,
             'drops': cf_mgr.model.drops,
             'num_cat': cf_mgr.model.n_classes,
             'class_to_idx': cf_mgr.model.class_to_idx,
             'state_dict': cf_mgr.model.state_dict(),
             'optimizer': cf_mgr.optimizer.state_dict(),
             'epochs': cf_mgr.epochs,
             'loss_function': cf_mgr.loss_function,
             'best_accuracy': cf_mgr.best_accuracy,
             'best_corrects': cf_mgr.best_corrects,
             'best_loss': cf_mgr.best_loss,
             'model': cf_mgr.model.state_dict(),
             'opt': cf_mgr.optimizer.state_dict(),
             }
    torch.save(state, chk_dir_name + '/' + 'checkpoint.pth.tar')

    #if HAS_CUDA:
    #    cf_mgr.model.cuda()


def load_classes(filename):

    import json

    with open(filename, 'r') as f:
        cat_to_name = json.load(f)

    num_cat = len(cat_to_name)
    print (f'Number of classes: {num_cat}')

    return cat_to_name


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array

        Note - the transformations are define in the data_transforms['predict'] dictionary as defined earlier

    '''
    # TODO: Process a PIL image for use in a PyTorch model

    image = data_transforms['test'](image).float()

    # Convert back to numpy array since this is what is requested
    image = image.numpy()

    return image


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()

    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.permute(1, 2, 0).numpy()

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)

    ax.imshow(image)

    return ax


def myconv(x):
    '''
    Function to convert class numbers to folder labels
    '''
    return idx_to_class[x]

def get_name(x):
    '''
    Function to convert folder lables to class names
    '''
    return cat_to_name[x]