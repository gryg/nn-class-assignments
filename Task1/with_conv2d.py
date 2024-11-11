import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple
import numpy as np

def get_conv_weight_and_bias(
        filter_size: Tuple[int, int],
        num_groups: int,
        input_channels: int,
        output_channels: int,
        bias: bool
) -> Tuple[torch.Tensor, torch.Tensor]:
    # assert that num_filters is divisible by num_groups
    assert input_channels % num_groups == 0, "input channels must be divisible by groups number"
    # assert that num_channels is divisible by num_groups
    assert output_channels % num_groups == 0, "output channels must be divisible by groups number"
    
    in_channels_per_group = input_channels // num_groups
    
    # Create weight matrix with shape (output_channels, in_channels_per_group, kh, kw)
    weight_matrix = torch.randn(output_channels, in_channels_per_group, *filter_size)
    
    if bias:
        bias_vector = torch.ones(output_channels)
    else:
        bias_vector = None
    return weight_matrix, bias_vector


class MyConvStub:
    def __init__(
            self,
            kernel_size: Tuple[int, int],
            num_groups: int,
            input_channels: int,
            output_channels: int,
            bias: bool,
            stride: int,
            dilation: int,
    ):
        self.weight, self.bias = get_conv_weight_and_bias(kernel_size, num_groups, input_channels, output_channels, bias)
        self.groups = num_groups
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        # Perform grouped convolution manually if necessary
        return F.conv2d(
            x,
            self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=0,
            dilation=self.dilation,
            groups=self.groups
        )

class MyFilterStub:
    def __init__(self, filter, input_channels):
        self.filter = filter  # Assign the provided filter to an instance attribute
        self.input_channels = input_channels

    def forward(self, x):
        # Ensure the filter is repeated across input channels for grouped convolution
        filter_expanded = self.filter[None, None, :, :].expand(self.input_channels, 1, *self.filter.shape)
        return F.conv2d(x, filter_expanded, groups=self.input_channels)

    