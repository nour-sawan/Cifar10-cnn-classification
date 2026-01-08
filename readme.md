# CIFAR-10 Image Classification using CNN

This project implements a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset.  
The workflow covers model training, validation, overfitting analysis, and performance improvement using EarlyStopping.

## Dataset
- CIFAR-10 dataset
- 10 image classes
- 32×32 RGB images

## Model Architecture
- Convolutional layers with ReLU activation
- MaxPooling layers
- Fully connected (Dense) layers
- Softmax applied at inference time for probability estimation

## Training Strategy
- Baseline CNN training
- Validation split from training data
- Overfitting detection using training vs validation loss
- EarlyStopping applied to reduce overfitting and restore best model weights

## Evaluation
- Model evaluated on a separate test set
- Accuracy and loss metrics reported
- Comparison between baseline model and EarlyStopping-enhanced model

## Prediction Example
- Sample predictions visualized
- Predicted class compared with ground-truth label
- Confidence scores obtained using Softmax probabilities

## Tools & Libraries
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib

## How to Run
1. Open the notebook in Google Colab or Jupyter Notebook
2. Run all cells from top to bottom
3. Training and evaluation results will be displayed automatically
