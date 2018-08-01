
import torch.nn as nn


class Composite_Classifier(nn.Module):
    """
    Create new model class which will combine pre trained network with classifier.  Can have multiple hidden
    layers, each with dropout

    Arguments:
        model_pt: The pretrained pytorch model to use as a backbone
        n_hid: A list containing the number of hidden layer neurons for each layer.  The number of layers is taken from list length
        drops: A list containing dropout probability for each hidden layer
        n_classes: the number of classes at the output of the model

    """

    def __init__(self, model_pt, n_hid, drops, n_classes):
        super(Composite_Classifier, self).__init__()
        self.drops=drops
        self.n_hid=n_hid
        self.n_classes = n_classes
        layers = list(model_pt.children())
        no_layers = len(layers)
        pt_layers = layers[:-1]
        self.cf_in = model_pt.fc.in_features
        # Create new model with the all of the pre-trained layers apart from the last
        self.model_pt = nn.Sequential(*(pt_layers))

        # Now define new layers and classifier
        self.activation = nn.ReLU()
        self.cf_n_hid = len(self.n_hid)

        self.cf_layers = nn.ModuleList([])

        # Define first layer of classifier making sure it has the same size as the last layer of the
        # pre-trained network

        if len(self.drops) > 0:
            dropout0 = nn.Dropout(p=self.drops[0])
            self.cf_layers.append(dropout0)
        new_layer = nn.Linear(self.cf_in, self.n_hid[0])
        self.cf_layers.append(new_layer)
        new_layer = nn.BatchNorm1d(self.n_hid[0])
        self.cf_layers.append(new_layer)
        new_layer = self.activation
        self.cf_layers.append(new_layer)

        # Add additional hidden layers if necessary
        for layer in range(1, self.cf_n_hid):
            if len(self.drops) > layer:
                dropout = nn.Dropout(p=self.drops[layer])
                self.cf_layers.append(dropout)
            new_layer = nn.Linear(n_hid[layer - 1], self.n_hid[layer])
            self.cf_layers.append(new_layer)
            new_layer = nn.BatchNorm1d(self.n_hid[layer])
            self.cf_layers.append(new_layer)
            new_layer = self.activation
            self.cf_layers.append(new_layer)

        # Add layer from last hidden layer to classifier
        to_cf = nn.Linear(n_hid[-1], self.n_classes)
        self.cf_layers.append(to_cf)
        new_layer = nn.BatchNorm1d(self.n_classes)
        self.cf_layers.append(new_layer)
        cf_layer_list=list(self.cf_layers)
        self.model_cf = nn.Sequential(*(cf_layer_list))

    def forward(self, input):
        # Feed data through pre-trained model
        x = self.model_pt(input)
        x = x.view(-1,self.cf_in)
        # now feed through classifier
        x = self.model_cf(x)
        return x