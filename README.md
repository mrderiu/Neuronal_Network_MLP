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

### A Note on ResNet18 (and the ResNet Family)

The name "ResNet18" sometimes shows up loosely attached to MLP code, but it actually refers to a specific, well-defined architecture that has nothing to do with the models above — it's worth understanding what it really is so the name isn't reused incorrectly.

**ResNet** (*Residual Network*) is a convolutional architecture introduced by He et al. (2015) for image classification. Its key contribution is the **residual (skip) connection**: instead of a block learning a direct mapping `H(x)`, it learns a residual `F(x)` and adds the original input back:

```
output = F(x) + x
```

This shortcut lets gradients flow directly through the addition during backpropagation, which solves the **vanishing gradient / degradation problem** that made very deep plain CNNs perform *worse* than shallower ones. Thanks to residual connections, networks with dozens or hundreds of layers became trainable.

**ResNet18** specifically:
* 18 refers to the number of layers with learnable weights (convolutional + fully connected), not to any hyperparameter you choose — it's a fixed, published architecture.
* Built from **basic blocks**: each block has two 3×3 convolutional layers, and the input is added to the block's output via the skip connection.
* It's the smallest/shallowest member of the ResNet family, commonly used as a lightweight baseline or for transfer learning on smaller datasets.

**Other ResNet variants**, for reference:

| Variant | Depth | Block type | Notes |
|---|---|---|---|
| ResNet18 | 18 layers | Basic block (2× conv 3×3) | Lightest, fastest, good baseline |
| ResNet34 | 34 layers | Basic block (2× conv 3×3) | Deeper version of the same block type |
| ResNet50 | 50 layers | Bottleneck block (1×1 → 3×3 → 1×1) | Switches to bottleneck blocks to keep the parameter count manageable despite the extra depth |
| ResNet101 | 101 layers | Bottleneck block | Higher capacity, more compute |
| ResNet152 | 152 layers | Bottleneck block | Deepest common variant, used when accuracy matters more than speed |

The **bottleneck block** (used from ResNet50 onward) first reduces the number of channels with a 1×1 convolution, applies the expensive 3×3 convolution on that smaller representation, and then restores the channel count with another 1×1 convolution. This keeps computation and memory reasonable even as depth increases.

**Why it doesn't apply to this project:** ResNet is a *convolutional* architecture designed for image (grid-structured) data. This repository works with **tabular data** (Adult Census Income), processed as flat feature vectors through `nn.Linear` layers, so there are no convolutions or spatial structure for ResNet's blocks to operate on. The two MLP variants used here (see below) are unrelated to ResNet — the residual **idea** (skip connections) can in principle be adapted to plain fully-connected networks, but that's a different, non-standard architecture and isn't what "ResNet18" refers to.

## Implementación en este repositorio

Este repositorio aplica la teoría anterior a un caso real de clasificación binaria: predecir si una persona gana más de 50K al año a partir del dataset **Adult Census Income**.

### Estructura del proyecto

```
main.py                 # Orquesta el pipeline completo (MLP)
main_xgboost.py          # Pipeline equivalente con XGBoost, para comparar
src/
├── config.py            # Rutas, columnas, hiperparámetros, selección de modelo
├── data_loader.py        # Carga del CSV y split train/validation/test
├── preprocessing.py      # Limpieza de valores y ColumnTransformer (impute + scale + one-hot)
├── models.py              # Arquitecturas del MLP + factory build_model()
├── early_stopping.py      # Implementación de Early Stopping
├── train.py               # Loop de entrenamiento (forward, loss, backward, optimizer)
└── evaluate.py            # Métricas sobre el conjunto de test
```

### Variantes de modelo (`src/models.py`)

Ambas arquitecturas devuelven **logits** (sin `Sigmoid` en la salida), porque `train.py` usa siempre `nn.BCEWithLogitsLoss`, que ya incorpora el sigmoid de forma numéricamente estable y permite ponderar la clase minoritaria con `pos_weight`. Mezclar un modelo con `Sigmoid` y `BCEWithLogitsLoss` aplicaría el sigmoid dos veces y rompería las probabilidades de salida, así que el proyecto mantiene esta regla de forma consistente en todas las variantes.

| Variante | Arquitectura | Cuándo usarla |
|---|---|---|
| `"simple"` | `Linear → ReLU` ×3 | Modelo base, sin regularización. Punto de partida o baseline. |
| `"regularized"` | `Linear → BatchNorm → ReLU → Dropout` ×2 | Añade normalización y regularización para reducir overfitting. Recomendada por defecto. |

La variante activa se elige en `config.MODEL_VARIANT` y se construye mediante la factory `build_model(name, input_dim)`, sin tener que modificar `main.py`.

### Manejo del desbalance de clases

La clase positiva (`income > 50K`) es minoritaria en el dataset. `train.py` calcula automáticamente `pos_weight = n_neg / n_pos` a partir del set de entrenamiento y lo pasa a `BCEWithLogitsLoss`, tal como se describe en la sección de *Imbalanced Data*.

### Early Stopping (`src/early_stopping.py`)

La clase `EarlyStopping` implementa exactamente el comportamiento descrito en la teoría:

- Monitoriza `val_loss` en cada epoch.
- Si no mejora durante `config.EARLY_STOPPING_PATIENCE` epochs consecutivos, detiene el entrenamiento.
- Guarda internamente los pesos del mejor epoch y los restaura al final, de forma que `train_model()` nunca devuelve un modelo peor que el mejor visto durante el entrenamiento.

### Ejecución

```bash
pip install -r requirements.txt
python main.py
```
