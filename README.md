# classifier
## Image classifier using transfer learning

This is a fairly standard implementation of an image classifier using transfer learning, PyTorch, and the PyTorch model zoo.

The models can be run using the supplied notebook or using the train.py and predict.py scripts.  In each case the underlying calculations are managed by the following files:

* composite_classifier.py: takes a pre-trained model from the pytorch model zoo and returns the same model with the last layer of the classifier removed and a new classifier added.  The number of layers in the new classifier can be controlled.  Each layer has dropout and batch norm added.  The dropout parameters can be controlled on a layer by layer basis

* solution_manager.py: manages the training, test and prediction using a model.  Allows for learning rate management and number of epochs

* data_manager.py: this contains a class that is used to link to the data and control the transforms.  It is provided as a convenience to make it easier to pass information to other classes

* utility.py:  Contains routines to save a checkpoint, load from a checkpoint, load an image, process an image, apply plot an image etc

## Training
to train a model from scratch using train.py use the following

python train.py path_to_image_folders --save_dir checkpoint_directory_name --arch resnet18 --gpu --n_hid "[512,256]" --drops "[0.2,0.2]" --learning_rate 0.01 --epochs 10

Note that n_hid and drops should both be lists where each entry is defines a layer.  The length of both must be the same.

To resume from a checkpoint using train.py use the following:

python train.py path_to_image_folders --resume checkpoint_dir_name --save_dir new_checkpoint_directory_name -epochs 10 --gpu

It is not necessary to define the architecture or hidden layers when resuming since these are loaded from the checkpoint.

Note that the resume does not seem to continue properly despite loading the optimiser state as well, this needs further investigation

## Prediction

To run an individual image file through the model saved at a checkpoint use the following command:

python predict.py path_to_image checkpoint_dir_name --topk 5 --category_names path_to_file --gpu

In this case the catagory_names option allows the user to specify the path to a JSON file containing mapping from catagory to name, where catagory is the folder name as a string and name is the description of the class as a string
