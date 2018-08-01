import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, transforms

class Solution_Manager():
    """
    This is a class to manage the training, test and prediction of a neural network.  All of the things it does
    could be done separately but this makes it less hassle
    """

    def __init__(self, model, loss_function, optimizer, scheduler, data, phases, HAS_CUDA=False):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.data = data
        self.datasets = datasets
        self.phases = phases
        self.HAS_CUDA = HAS_CUDA
        self.best_accuracy = 0.0
        self.best_corrects = 0
        self.best_loss = 0.0
        self.best_model_wts = []
        self.optimizer.best_model = []
        self.best_model_opt = []
        self.epochs = 0


    def to_np(self, x):
        if self.HAS_CUDA:
            return x.data.cpu().numpy()
        else:
            return x.data.numpy()

    def load_model_weights(self, filename):
        # needs mechanism putting in place to check file exists etc and error recovery
        self.model.load_state_dict(torch.load(filename))
        if self.HAS_CUDA:
            self.model.cuda()
        return

    def train(self, epochs):
        TRAIN = 'train'
        VAL = 'valid'
        phases=[TRAIN, VAL]

        # Create a copy of the weights so that we can return to them if we
        # go past the best solution
        self.best_model_wts = copy.deepcopy(self.model.state_dict())
        for epoch in range(self.epochs, self.epochs + epochs):
            print(f'Epoch: {epoch+1} / {self.epochs+epochs}')
            for phase in [TRAIN, VAL]:
                if phase == TRAIN:
                    self.scheduler.step()
                    self.model.train
                else:
                    self.model.eval()
                # Initialise counters
                cumulative_loss = 0.0
                cumulative_correct = 0

                # work out way through the data a batch at a time
                for x, y in self.data.dataloaders[phase]:
                    x = V(x)
                    y = V(y)
                    if self.HAS_CUDA:
                        x = x.cuda()
                        y = y.cuda()
                    self.optimizer.zero_grad()
                    pred = self.model(x)
                    _, pred_class = torch.topk(pred.data, 1, dim=1)
                    loss = self.loss_function(pred, y)

                    if phase == TRAIN:
                        loss.backward()
                        self.optimizer.step()

                    cumulative_loss += loss.data * x.size(0)
                    cumulative_correct += torch.sum(pred_class[:, 0] == y.data)

                epoch_loss = cumulative_loss / self.data.dataset_sizes[phase]
                if self.HAS_CUDA:
                    epoch_loss = epoch_loss.cpu()
                epoch_acc = float(cumulative_correct) / self.data.dataset_sizes[phase]
                print(f'Phase: {phase}, loss = {float(epoch_loss):.4f}, accuracy = {epoch_acc:.4f}')

            # Make a copy of the model if it is the most accurate
            if phase == VAL and epoch_acc > self.best_accuracy:
                self.best_accuracy = epoch_acc
                epoch_loss = float(epoch_loss)
                self.best_loss = epoch_loss
                self.best_model_wts = copy.deepcopy(self.model.state_dict())
                self.best_model_opt = copy.deepcopy(self.optimizer.state_dict())

        print('Finished training')
        print(f'Best accuracy: {self.best_accuracy:.4f}    Best loss: {self.best_loss:.4f}')

        # Load best weights
        self.model.load_state_dict(self.best_model_wts)
        # self.optimizer.load_state_dict(self.optimizer.best_model)
        self.epochs = epochs
        return

    def test_with_dl(self):
        phase = 'test'

        # Initialise counters
        cumulative_loss = 0.0
        cumulative_correct = 0
        # Switch model to eval mode
        self.model.eval()
        # work out way through the data a batch at a time
        for x, y in self.data.dataloaders[phase]:
            x = V(x)
            y = V(y)
            if self.HAS_CUDA:
                x = x.cuda()
                y = y.cuda()

            pred = self.model(x)
            _, pred_class = torch.topk(pred.data, 1, dim=1)
            loss = self.loss_function(pred, y)

            cumulative_loss += loss.data * x.size(0)
            cumulative_correct += torch.sum(pred_class[:, 0] == y.data)
        epoch_loss = cumulative_loss / self.data.dataset_sizes[phase]
        if self.HAS_CUDA:
            epoch_loss = epoch_loss.cpu()
        epoch_loss = float(epoch_loss)
        epoch_acc = float(cumulative_correct) / self.data.dataset_sizes[phase]
        print(f'Phase: {phase}, loss = {epoch_loss:.4f}, accuracy = {epoch_acc:.4f}')
        return

    def predict(self, images):
        self.model.eval()
        # Wrap the tensor as a variable
        images = V(images)
        if self.HAS_CUDA:
            images = images.cuda()
        pred = self.model(images)
        if self.HAS_CUDA:
            pred = pred.cpu()
        pred = F.softmax(pred, dim=1)

        # Note = could apply a softmax here to give probabilities but not strictly necessary to identify
        # most probable classes

        return pred.data
