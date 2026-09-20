# Ensemble CNN for Fashion-MNIST

## Objective

The objective of this project is to develop an ensemble of Convolutional Neural Networks (CNNs) for classifying images from the Fashion-MNIST dataset and compare its performance with a single CNN model.

## Dataset

The Fashion-MNIST dataset is provided through Keras and contains grayscale images of clothing items.

- Image size: 28 × 28 pixels
- Number of classes: 10
- Image type: Grayscale

For this experiment, only the first 50 training records and first 50 test records were used.

## Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the Fashion-MNIST dataset using `keras.datasets.fashion_mnist.load_data()`.
2. Selected the first 50 training and test records.
3. Normalized pixel values from 0–255 to the range 0–1.
4. Reshaped the images from `(28, 28)` to `(28, 28, 1)`.
5. Split the training data into training and validation sets using `train_test_split`.

The training data was divided into:

- 40 training samples
- 10 validation samples

## CNN Architecture

Each CNN model uses the following architecture:

```text
Input Image (28 × 28 × 1)
        ↓
Conv2D
32 Filters
3 × 3 Kernel
ReLU Activation
        ↓
MaxPooling2D
2 × 2 Pool
        ↓
Flatten
        ↓
Dense
10 Neurons
Softmax Activation
