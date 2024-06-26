# Medical BloodCell Classifiers

## Project Overview

This project focuses on implementing classifier models via supervised learning to correctly classify images from the BloodMNIST dataset. The dataset is part of the MedMNIST collection, designed to match the shape of the original digits MNIST dataset. Specifically, this project aims to predict the classification of blood cell images using various machine learning models.

## Dataset

The BloodMNIST dataset contains health-related images, each sample being a 28x28 RGB image. The dataset is already split into training, validation, and test sets, but the images are not normalized. Pre-processing steps, such as normalization, are necessary to prepare the data for model training.

## Objectives

1. **Model Selection and Justification**:
   - Choose at least 4 different classifier architectures.
   - Provide a text description and justification for the selected models.

2. **Model Training and Evaluation**:
   - Train the selected models on the BloodMNIST dataset.
   - Compare their performance using appropriate evaluation metrics.

## Classifier Architectures

The project explores a variety of classifier models to ensure a diverse approach to the problem. The chosen architectures include:

1. **Logistic Regression**:
   - A simple baseline model for comparison.

2. **Fully Connected Neural Network**:
   - A multi-layer perceptron with one or more hidden layers.

3. **Convolutional Neural Network (CNN)**:
   - Includes convolutional layers, pooling layers, and fully connected layers, suitable for image data.

4. **MobileNet**:
   - A lightweight, efficient convolutional neural network designed for mobile and embedded vision applications.

## Implementation

### Data Loading and Pre-processing

The dataset is loaded and split into training, validation, and test sets. The images are normalized to ensure optimal performance of the classifiers.

### Model Training

Each model is trained using the training set, validated on the validation set, and evaluated on the test set using metrics such as accuracy, precision, recall, and F1-score.

### Performance Comparison

The performance of all models is compared and visualized using confusion matrices and other relevant metrics.

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/Vivek-Tate/Medical-BloodCell-Classifier.git
   ```

## Dependencies

- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- TensorFlow/Keras or PyTorch
- Matplotlib/Seaborn

## Acknowledgements

- The MedMNIST dataset creators for providing the data. The BloodMNIST dataset can be downloaded from [Zenodo](https://zenodo.org/record/6496656/files/bloodmnist.npz?download=1).
