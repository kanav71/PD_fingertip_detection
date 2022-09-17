# PD_fingertip_detection
Codes for finding coordinates of thumb & fingertip of hand across images 

### Train-test-validation split
script_for_train_test_valid_pics_split.ipynb : Script splits the videos into frames and shuffles them across train-test-validation folders

### Fingertip detection

#### Pre-process folder
augmentation.py : Contains all the data augmentation steps
generator.py : Script for generation of training and validation images & ground truth labels for training the models. 

#### Utils folder
history.py: Code for generating the loss-epoch curve 

#### Net folder
Contains codes for calling different model architectures along with the configuration of top FC layers. 

#### Weights
Folder for saving the trained model weights.

#### Training the model
train.py: Code for initiating the model training. 
hyperparam_train.py : Code for hyperparameter tuning with early stopping.
performance.py: Code for evaluating the performance o the trained model.
Script for plotting & checking predicted labels.ipynb : Code for using the trained model to predict and plot the fingertip coordinates on any image, alongwith the ground truth labels. 

