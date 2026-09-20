# Ensemble CNN for Fashion-MNIST

## Objective

The objective of this project is to develop an ensemble of Convolutional Neural Networks (CNNs) to classify images from the Fashion-MNIST dataset and compare the performance of the ensemble with a single CNN model.

The ensemble consists of five CNN models trained using bootstrap sampling. The prediction probabilities from all five models are averaged to produce the final ensemble prediction.

---

## Dataset

The Fashion-MNIST dataset is a collection of grayscale images of clothing and fashion items.

The dataset contains 10 different classes:

| Label | Class |
|---:|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

### Dataset Information

- Image size: `28 × 28` pixels
- Image type: Grayscale
- Number of classes: 10
- Pixel value range: 0–255

For this experiment, only the first **50 training records** and first **50 test records** were used as specified in the task.

---

## Data Loading

The Fashion-MNIST dataset was loaded using Keras:

```python
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
