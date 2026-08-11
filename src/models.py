import torch
import torch.nn as nn


def mlp_model_simple(input_dim: int) -> nn.Module:
    """
    Baseline MLP for binary classification: Linear + ReLU layers only,
    no regularization. Returns raw logits (no Sigmoid in the output),
    meant to be used together with nn.BCEWithLogitsLoss.
    """
    model = nn.Sequential(
        # in_features=input_dim  -> number of preprocessed input features per sample
        # out_features=64        -> number of neurons in this layer / size of the output vector
        nn.Linear(input_dim, 64),
        nn.ReLU(),  # non-linear activation: keeps positive values, zeroes out negative ones

        # in_features=64  -> must match the previous layer's out_features
        # out_features=32 -> compresses the representation into 32 features
        nn.Linear(64, 32),
        nn.ReLU(),

        # in_features=32 -> must match the previous layer's out_features
        # out_features=8 -> further compression into 8 features
        nn.Linear(32, 8),
        nn.ReLU(),

        # in_features=8  -> must match the previous layer's out_features
        # out_features=1 -> single output logit for binary classification
        nn.Linear(8, 1),
    )
    return model


def mlp_model_regularized(input_dim: int) -> nn.Module:
    """
    Same base architecture as mlp_model_simple, with BatchNorm1d and
    Dropout added after every hidden layer to improve generalization.

    Pattern: Linear -> BatchNorm -> ReLU -> Dropout

    Returns raw logits (no Sigmoid), for use with nn.BCEWithLogitsLoss.
    """
    model = nn.Sequential(
        # in_features=input_dim, out_features=64 -> same meaning as in mlp_model_simple
        nn.Linear(input_dim, 64),
        # num_features=64 -> must match the previous layer's out_features;
        # normalizes each of the 64 activations across the batch (mean 0, std 1),
        # then rescales them with two learnable parameters (gamma, beta)
        nn.BatchNorm1d(64),
        nn.ReLU(),
        # p=0.3 -> probability of randomly zeroing each neuron's output
        # during training (30% of neurons dropped on every forward pass)
        nn.Dropout(0.3),

        nn.Linear(64, 32),
        nn.BatchNorm1d(32),  # num_features=32 -> matches the 32 outputs of the previous layer
        nn.ReLU(),
        nn.Dropout(0.2),  # p=0.2 -> lighter dropout, since this layer is already smaller

        # in_features=32, out_features=1 -> single output logit
        nn.Linear(32, 1),  # no Sigmoid: the model outputs logits
    )
    return model


class ResidualBlock(nn.Module):
    """
    Fully-connected residual block, inspired by ResNet's skip connections
    (see the README section "A Note on ResNet18"), adapted here for
    tabular data instead of images.

    A convolutional ResNet basic block uses two 3x3 convolutions and adds
    the block's input to its output. This block does the same thing with
    two Linear layers of equal width, so input and output shapes match
    and can be summed directly:

        output = F(x) + x

    where F(x) is what the two Linear layers learn, and x is the
    original input to the block (the "skip connection").
    """

    def __init__(self, width: int, dropout: float = 0.2):
        """
        width   : number of input AND output features of the block. It
                  must stay the same on both sides, because the skip
                  connection adds the raw input x to F(x), and that
                  addition only works if both tensors have the same shape.
        dropout : probability of zeroing a neuron's output, applied once
                  per block, after the residual addition and activation.
        """
        super().__init__()
        self.block = nn.Sequential(
            # in_features=width, out_features=width -> keeps dimensionality
            # unchanged so the output can later be added to the input x
            nn.Linear(width, width),
            nn.BatchNorm1d(width),  # num_features=width
            nn.ReLU(),
            # second Linear layer of the block, same width as the first
            nn.Linear(width, width),
            nn.BatchNorm1d(width),
            # no ReLU here: the activation is applied AFTER the residual
            # addition below (this mirrors how ResNet basic blocks work)
        )
        self.relu = nn.ReLU()  # activation applied to F(x) + x, not to F(x) alone
        self.dropout = nn.Dropout(dropout)  # p=dropout -> applied after the residual addition

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x            # keep a reference to the original input (the skip connection)
        out = self.block(x)     # F(x): the residual learned by this block
        out = out + identity    # output = F(x) + x
        out = self.relu(out)
        out = self.dropout(out)
        return out


# Registry of available architectures. Lets you select a model by name
# (e.g. from config.py) without having to edit main.py.
MODEL_REGISTRY = {
    "simple": mlp_model_simple,
    "regularized": mlp_model_regularized,
}


def build_model(name: str, input_dim: int) -> nn.Module:
    """
    Factory: builds a model from its name in MODEL_REGISTRY.

    name      : key registered in MODEL_REGISTRY ("simple", "regularized"
                or "resnet").
    input_dim : number of input features, forwarded to the chosen
                model-building function.
    """
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Modelo '{name}' no reconocido. Opciones disponibles: "
            f"{list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](input_dim)
