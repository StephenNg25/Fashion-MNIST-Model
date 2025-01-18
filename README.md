# Fashion-MNIST-Classification-Model

This project builds a deep learning model to classify images in the Fashion MNIST dataset. The dataset includes grayscale images of clothing items and their categories, such as t-shirts, pants, and shoes. The goal is to train a neural network that accurately predicts the category of a given image.

## Workflow Breakdown

### 1. Fetch the Dataset

- The Fashion MNIST dataset is fetched using the Keras library.
- The dataset consists of two subsets:
    - Training Data: 60,000 images
    - Testing Data: 10,000 images
- Each subset includes:
    - An array of input images, where each image is a 28x28 grayscale pixel grid.
    - An array of category labels (e.g., 0, 1, 2) representing the class of each image.

### 2. Preprocessing

- Normalization: Pixel values in the images, initially ranging from 0 to 255, are scaled to a range of 0 to 1. This to improve training stability and ensures faster convergence.
- Reshaping: The input images, originally in a 2D format (28x28), are prepared for the model by flattening them into a 1D array of 784 pixels. This to make them compatible with the neural network.


---

### 3. Visualizing the Data
The dataset contains 10 classes, each representing a category of fashion items:
0: T-shirt/top
1: Trouser
2: Pullover
3: Dress
4: Coat
5: Sandal
6: Shirt
7: Sneaker
8: Bag
9: Ankle boot
Images are visualized during preprocessing to ensure the input data is structured correctly.

---

### 4. Neural Network Architecture

The model is constructed using the Keras Sequential API, which allows layers to be added step by step.

Input Layer:

A Flatten layer reshapes the 28x28 image grid into a 1D array of 784 pixels. This transformation is necessary because Dense layers can only accept 1D inputs.
Hidden Layers:

The network includes two hidden Dense (fully connected) layers with:
300 neurons in the first layer.
100 neurons in the second layer.
Each neuron in these layers applies the ReLU (Rectified Linear Unit) activation function. This introduces non-linearity, allowing the model to learn complex patterns in the data.
Output Layer:

The final Dense layer has 10 neurons, each representing a class (e.g., T-shirt, Trouser).
A softmax activation function converts the output into probabilities for each class. The highest probability indicates the predicted category.


### 5. Model Compilation

The model is configured with three key components:

Loss Function: Measures the difference between the model’s predictions and the actual labels. This project uses sparse_categorical_crossentropy, which is suitable for multi-class classification with integer labels.
Optimizer: Controls how the model’s weights are updated during training. The Stochastic Gradient Descent (SGD) optimizer is used here.
Metrics: Accuracy is tracked during training to evaluate the model’s performance.

### 6. Training the Model

The model is trained using the fit function:
Training Data: The input images and their corresponding labels are passed to the model in batches.
Validation Data: A separate dataset (X_valid, y_valid) is used to evaluate the model's performance after each epoch (a complete pass through the training data).
The training process involves:
Forward Pass: The model computes predictions by passing input data through its layers.
Loss Computation: The difference between predictions and true labels is calculated.
Backward Pass: Gradients of the loss are computed with respect to the model’s weights.
Weight Updates: The optimizer updates the weights to minimize the loss.


### 7. Monitoring Results

During training, the following metrics are tracked:

- Training Loss and Accuracy: Indicate how well the model is learning the training data.
- Validation Loss and Accuracy: Show how well the model performs on unseen data.

Example results:
- Training Accuracy: ~95.8%
- Validation Accuracy: ~89.5%

---

### 8. Testing and Prediction

After training, the model is tested on unseen validation images:

- Predictions:
  - For each input image, the model outputs probabilities for all 10 categories.
  - The class with the highest probability is selected as the predicted label.
- Visualization:
  - Images from the validation set are displayed alongside their predicted and true labels.

---

## Results

The following plots summarize the training process:

1. Accuracy Plot:
   - Displays how training and validation accuracy change over epochs.
2. Loss Plot:
   - Tracks the training and validation loss, helping identify potential overfitting.
