
"""
This module defines a Deep Kernel Learning (DKL) model for regression tasks,
leveraging GPyTorch and BoTorch libraries. The DKL model uses a deep neural network
as part of a kernel function to capture complex relationships in the data.
"""

from typing import Optional
 
import gpytorch
import torch
import torch.nn as nn
from botorch.models import SingleTaskGP
from botorch.models.transforms.input import InputTransform
from botorch.models.transforms.outcome import OutcomeTransform
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from torch import Tensor


class FeatureExtractor(nn.Module):
    """
    A simple feedforward neural network used as a feature extractor in the DKL model.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: Optional[list] = None):
        """
        Initializes the neural network.
        Defaults to 
            input_dim -> 500 -> 50 -> output_dim, if hidden_dims is not provided.
        LeakyReLU activations are used between layers.
        
        Parameters:
        -----------
            input_dim: int
                The dimensionality of the input features.
            output_dim: int
                The dimensionality of the output features.
            hidden_dims: list[int] or None
                A list of integers specifying the number of neurons in each hidden layer.
        """
        super(FeatureExtractor, self).__init__()
        if hidden_dims is None:
            hidden_dims = [500, 50]
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LeakyReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x)
    

class DeepKernel(gpytorch.kernels.Kernel):
    """
    A custom kernel that incorporates a neural network for feature extraction.
    based on Deep Kernel Learning (DKL).
    """
    has_lengthscale = False  # This kernel does not have a lengthscale parameter. The base kernel does.
    def __init__(self, feature_extractor: FeatureExtractor, base_kernel:Optional[gpytorch.kernels.Kernel] = None):
        """
        Initializes the Deep Kernel.

        Parameters:
        -----------
            feature_extractor: FeatureExtractor
                The neural network used for feature extraction.
            base_kernel: gpytorch.kernels.Kernel or None
                The base kernel to use after feature extraction. If None, defaults to RBFKernel.
        """
        super(DeepKernel, self).__init__()
        self.feature_extractor = feature_extractor
        if base_kernel is not None:
            self.base_kernel = base_kernel
        else:
            self.base_kernel = ScaleKernel(RBFKernel())
    
    def forward(self, x1: Tensor, x2: Tensor, **kwargs) -> Tensor:
        """
        Computes the k(g(x1), g(x2)).

        Parameters:
        -----------
            x1: Tensor
                The first set of input features.
            x2: Tensor
                The second set of input features.

        Returns:
        --------
            Tensor
                The computed kernel matrix.
        """
        # Pass inputs through the neural network to get feature representations
        features_x1 = self.feature_extractor(x1)
        features_x2 = self.feature_extractor(x2)
        
        # Compute the base kernel on the transformed features
        return self.base_kernel(features_x1, features_x2)