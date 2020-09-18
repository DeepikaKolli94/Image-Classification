# Image-Classification

CIFAR-10 dataset consists of 60000 32x32 colour images with 10 classes. There are around 50000 training images and 10000 test images. Each image has 3 channels representing the RGB channels. simple Softmax, 2-layer Neural network, 1 Convolution, and 3 Convolution layer network are implemented in this project using PyTorch. These models are attached to the ResNet18 model by removing the last fully-connected layer. Extensive hyperparameter tuning has been performed to obtain a better accuracy. Shell scripts contain the best hyperparameter values that produced these results.

# Code Files
train.py, cifar10.py and cifar10.pyc, Models: softmax.py, twolayernn.py, convnet.py, mymodel.py, Shell scripts: run_softmax.sh, run_twolayernn.sh, run_convnet.sh, run_mymodel.sh

# Parameters:
lr: learning rate, momentum: SGD momentum, weight-decay: Weight decay hyperparameter, batch-size: Input batch size for training, epochs: Number of epochs to train, model ('softmax', 'convnet', 'twolayernn', 'mymodel'): which model to train/evaluate, hidden_dim: number of hidden features/activations, kernel-size: size of convolution kernels/filters, finetuning: set requires_grad=False for non-updatable pretrained resnet18 model

# Steps to run the model:
1) Set desired parameters in the model's shell script
2) Open bash shell
3) Give permissions for execution using "chmod +x *.sh" 
4) Run the desired models shell script
5) log files and model with .pt extension are created automatically 
