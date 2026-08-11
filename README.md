## Neuronal Network

### Introduction to Deep Learning with PyTorch

Deep Learning is a branch of Machine Learning that focuses on training neural networks with multiple layers to learn complex patterns from data. These networks are inspired by the way biological neurons communicate, although they are much simpler mathematically. The most basic type of neural network is the Multilayer Perceptron (MLP). An MLP consists of several layers of artificial neurons (also called perceptrons) connected together. Each neuron receives one or more inputs, multiplies them by learnable weights, adds a bias term, and applies an activation function to produce an output.

Neurons are organized into layers:
* **Input Layer**: Receives the input features from the dataset.
* **Hidden Layers**: Perform intermediate computations and learn complex relationships.
* **Output Layer**: Produces the final prediction.

The arrangement of these layers is called the network architecture or network topology. Once the architecture is defined, the model must be trained using an optimization algorithm. When a neural network is created, it does not know how to solve the problem. All of its weights and biases are initialized with random values, so its predictions are essentially random. The goal of training is to find the set of weights that minimizes the prediction error and allows the model to make accurate predictions. To measure how well the model performs, a loss function is used. The loss function compares the model's predictions with the true labels and returns a numerical value representing the prediction error. A high loss indicates poor predictions, while a low loss indicates that the model is making accurate predictions. Once the loss has been computed, the backpropagation algorithm calculates the gradient of the loss with respect to every trainable parameter in the network. These gradients indicate how each weight should be adjusted to reduce the prediction error.

The optimization algorithm uses these gradients to update the weights after each training step. Instead of searching randomly for better parameters, the optimizer follows the direction that decreases the loss. This process is repeated many times until the model converges to a set of weights that performs well on the training data.

The most common optimization algorithms are:

* **SGD (Stochastic Gradient Descent)**: The simplest optimizer. It updates weights using only the current gradient.
* **Momentum**: Extends SGD by remembering previous updates, helping the optimization move faster and avoid oscillations.
* **RMSprop**: Adapts the learning rate for each parameter based on recent gradients.
* **Adam (Adaptive Moment Estimation)**: Combines the ideas of Momentum and RMSprop. It adapts the learning rate for each parameter and is one of the most popular optimizers because it usually converges faster and requires less tuning.

The overall training process can be summarized as follows:

* Perform a forward pass to compute the model's predictions.
* Compute the loss by comparing the predictions with the true labels.
* Use backpropagation to calculate the gradients of the loss.
* Apply the optimizer to update the model's weights.
* Repeat this process for every mini-batch and every epoch until the model reaches satisfactory performance.

Without an optimization algorithm, the neural network would never improve beyond its initial random predictions, making it impossible for the model to learn meaningful patterns from the data.

### Dataset Inputs and Outputs

Each sample in the dataset consists of:

* Input features (predictors): Numerical variables used to make predictions.
* Target (label): The expected output.

### Building a Neural Network in PyTorch

PyTorch provides the torch.nn module for creating neural networks. A simple Multilayer Perceptron can be defined as follows:
```
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(8, 12),
    nn.ReLU(),

    nn.Linear(12, 8),
    nn.ReLU(),

    nn.Linear(8, 1),
    nn.Sigmoid()
)
```
This model contains:

* One input layer
* Two hidden layers
* One output layer

nn.Linear implements a fully connected (dense) linear layer. It performs a linear transformation on the input data by multiplying the input vector by a learnable weight matrix and optionally adding a bias term. Mathematically, the operation is:

```
y = Wx + b

where:

x is the input vector.
W is the matrix of learnable weights.
b is the learnable bias.
y is the output vector.
```
The syntax is:

nn.Linear(in_features, out_features)

The numbers in each nn.Linear(in_features, out_features) layer define the shape of the neural network.

* The first number (in_features) is the number of values the layer receives as input.
* The second number (out_features) is the number of values (or neurons) the layer produces as output.

The following are the details of the different layers:

* First Layer: nn.Linear(8, 12) --> Receives 8 input values and produce 12 output values. This means that each input sample is represented as a vector of 8 numbers, and the layer transforms it into a new vector containing 12 values.
* Second Layer: nn.Linear(12, 8) --> The previous layer produced 12 values, so this layer must accept exactly 12 inputs. It transforms those 12 values into a new representation containing 8 values.
* Output Layer: nn.Linear(8, 1) --> The final hidden layer produces 8 values. The output layer combines these into a single output value.

The numbers determine the capacity of the network. For example, nn.Linear(8, 12) does not mean that the dataset has 12 features. It simply means that the layer receives 8 numbers and computes a richer internal representation using 12 neurons. Likewise, nn.Linear(12, 8) compresses those 12 learned features into 8 new features. The hidden layers are learning increasingly useful representations of the input data.

The numbers 12 and 8 correspond to the number of neurons in the hidden layers. A neuron computes output = activation(weight × input + bias). Each neuron learns a different pattern from the input data.

For example:

Hidden Layer (12 neurons)
```
Neuron 1 → Pattern A
Neuron 2 → Pattern B
Neuron 3 → Pattern C
...
Neuron 12 → Pattern L
```
The meaning of these patterns is not predefined. They are discovered automatically during training.

There is no universal rule for selecting the number of neurons. The architecture is a design choice made by the developer. For example, all of the following are valid:
```
nn.Linear(8, 16)
nn.Linear(8, 32)
nn.Linear(8, 64)
nn.Linear(8, 128)
```
Larger layers can learn more complex patterns, but they also require more data, more computation, and increase the risk of overfitting.

nn.ReLU (Rectified Linear Unit) is an activation function that introduces non-linearity into the neural network. If the input is positive, the output remains unchanged. If the input is negative, the output becomes zero.

```
Input	 Output
 -3	     0
 -1	     0
  0	     0
  2	     2
  5	     5
```
Without an activation function such as ReLU, stacking multiple linear layers would still produce a single linear transformation. ReLU allows the network to learn complex, non-linear relationships in the data.

nn.Sigmoid is an activation function commonly used in the output layer of binary classification models. It transforms any real-valued input into a value between 0 and 1. The output can be interpreted as a probability:

* Values close to 0 indicate a low probability of the positive class.
* Values close to 1 indicate a high probability of the positive class.

```
Raw Output	Sigmoid Output
   -4	             0.018
   -1	             0.269
    0	             0.500
    2	             0.881
    5	             0.993
```
For this reason, Sigmoid is typically used as the final activation function in binary classification networks, where the model predicts the probability that an input belongs to one of two possible classes.

The previous model is an excellent starting point for learning how neural networks work. However, building deep learning models is an incremental process. As you become more familiar with PyTorch, you will gradually improve both the architecture and the training procedure. The progression is not simply about adding more layers. Instead, it is about understanding how to train, evaluate, regularize, and adapt a neural network to different types of problems.

#### Replace Sigmoid with BCEWithLogitsLoss
A common improvement in binary classification is to remove the Sigmoid activation from the model and use BCEWithLogitsLoss during training.

Instead of
```
model = nn.Sequential(
    nn.Linear(8, 12),
    nn.ReLU(),
    nn.Linear(12, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid()
)
```
a more modern implementation is
```
model = nn.Sequential(
    nn.Linear(8, 12),
    nn.ReLU(),
    nn.Linear(12, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)

loss_fn = nn.BCEWithLogitsLoss()
```
BCEWithLogitsLoss combines the Sigmoid activation and Binary Cross Entropy into a single operation that is more numerically stable and is the recommended approach in PyTorch.

#### More Features

* **Dropout**
As neural networks become larger, they may memorize the training data instead of learning general patterns. Dropout is a regularization technique that randomly disables a percentage of neurons during training.

Example:
```
nn.Linear(32, 16),
nn.ReLU(),
nn.Dropout(0.2)
```
This forces the network to learn more robust representations and reduces overfitting.

* **Normalize Activations**
BatchNorm1d normalizes the outputs of a layer during training. A common pattern is:
```
Linear --> BatchNorm --> ReLU --> Dropout
```
Batch Normalization often stabilizes training and allows the optimizer to converge faster.

* **Dataset and DataLoader**
Instead of manually creating mini-batches with array slicing, PyTorch provides the Dataset and DataLoader classes.

DataLoader automatically:

* creates mini-batches,
* shuffles the training data,
* loads batches efficiently,
* simplifies the training loop.

These classes are the standard way of feeding data into neural networks.

* **Imbalanced Data**
Many real-world datasets contain significantly more samples from one class than the other. For binary classification problems, PyTorch provides the pos_weight parameter in BCEWithLogitsLoss to give more importance to the minority class during training. This is particularly useful for applications such as fraud detection, anomaly detection, and medical diagnosis.

* **Early Stopping**
Training for too many epochs may cause the model to overfit the training data. Early stopping monitors the validation loss and automatically stops training when the model no longer improves. It also saves the best-performing version of the model.


### Project Structure

```
main.py                  # Orchestrates the complete pipeline: trains, compares,
                         # and saves the best model for production
src/
├── config.py            # Paths, columns, hyperparameters, and model selection
├── data_loader.py       # CSV loading and train/validation/test split
├── preprocessing.py     # Data cleaning and ColumnTransformer
│                        # (imputation + scaling + one-hot encoding)
├── models.py            # MLP architectures + build_model() factory
├── early_stopping.py    # Early Stopping implementation
├── train.py             # Training loop (forward, loss, backward, optimizer)
├── evaluate.py          # Validation and test evaluation metrics
└── experiment_log.py    # Logs each experiment run to logs/experiments.csv
predict.py               # Production inference using the saved winning model
```

### Model Variants (`src/models.py`)

Both architectures return **logits** (without applying `Sigmoid` to the output), because `train.py` always uses `nn.BCEWithLogitsLoss`. This loss function already incorporates the sigmoid operation in a numerically stable way and also allows the minority class to be weighted using `pos_weight`.

Combining a model that already applies `Sigmoid` with `BCEWithLogitsLoss` would apply the sigmoid function twice and produce incorrect output probabilities. For this reason, the project consistently follows the logits-only approach across all model variants.

| Variant | Architecture | When to Use |
| --- | --- | --- |
| `"simple"` | `Linear → ReLU` ×3 | Baseline model without regularization. Suitable as a starting point or reference model. |
| `"regularized"` | `Linear → BatchNorm → ReLU → Dropout` ×2 | Adds normalization and regularization to reduce overfitting. Recommended as the default architecture. |

The active variant can be configured through `config.MODEL_VARIANT` and is instantiated using the `build_model(name, input_dim)` factory, without requiring any changes to `main.py`.

### Architecture Comparison and Best Model Selection

`main.py` does not train a single architecture. Instead, it iterates through **all** model variants registered in `MODEL_REGISTRY` (`src/models.py`), trains each architecture independently, and compares their performance.

This design makes the project easily extensible: adding a new architecture to `MODEL_REGISTRY` automatically includes it in the comparison process without requiring any changes to `main.py`.

The winning model is selected based on its **validation F1 score**, rather than accuracy or test performance:

- **F1 instead of accuracy** because the positive class (`income > 50K`) is underrepresented in the dataset. Accuracy can therefore be misleading when evaluating an imbalanced classification problem.
- **Validation instead of test** because the test set is reserved exclusively for the final evaluation of the selected model. Using test performance for model selection would introduce information leakage from the test set into the model selection process.

### Experiment Tracking (`src/experiment_log.py`)

Each trained architecture adds a new row to:

`logs/experiments.csv`

The log contains information such as:

- Timestamp
- Model variant
- Validation accuracy
- Validation precision
- Validation recall
- Validation F1
- Test accuracy
- Test precision
- Test recall
- Test F1

The file is created automatically if it does not already exist.

Each execution of `main.py` appends new rows without overwriting previous results, providing a historical record of all experiment runs.

### Production Artifacts and `predict.py`

Once all architectures have been compared, `main.py` saves everything required to perform inference with the winning model under:

`models/production/`

The following artifacts are generated:

- `model.pt` — trained model weights (`state_dict`).
- `preprocessor.joblib` — the `ColumnTransformer` fitted on the training dataset.
- `metadata.json` — metadata describing the winning architecture and the `input_dim` used to construct it. This information is required to reconstruct the neural network before loading its weights.

`predict.py` loads these production artifacts and exposes a `predict(df)` function that can generate predictions for new, unseen data without retraining the model.

```bash
python predict.py


