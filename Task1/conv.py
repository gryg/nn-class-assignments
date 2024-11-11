import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple, Optional

def manual_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: int = 1,
    dilation: int = 1,
    groups: int = 1
) -> torch.Tensor:
    batch_size, in_channels, in_height, in_width = x.shape
    out_channels, in_channels_per_group, kernel_height, kernel_width = weight.shape
    
    # Validate input dimensions
    assert in_channels == in_channels_per_group * groups, "Input channels must match weight dimensions"
    
    # Calculate output dimensions
    out_height = (in_height - (kernel_height - 1) * dilation - 1) // stride + 1
    out_width = (in_width - (kernel_width - 1) * dilation - 1) // stride + 1
    
    # Split input into groups
    x_grouped = x.view(batch_size * groups, in_channels // groups, in_height, in_width)
    
    # Unfold the input into patches
    x_unfolded = F.unfold(
        x_grouped,
        kernel_size=(kernel_height, kernel_width),
        stride=stride,
        dilation=dilation
    )
    
    # Reshape weight for grouped convolution
    weight_grouped = weight.view(groups, out_channels // groups, in_channels_per_group * kernel_height * kernel_width)
    
    # Reshape unfolded input for matrix multiplication
    x_unfolded = x_unfolded.view(batch_size, groups, in_channels_per_group * kernel_height * kernel_width, -1)
    
    # Perform convolution using matrix multiplication for each group
    out = []
    for g in range(groups):
        # Matrix multiplication for current group
        group_out = torch.matmul(weight_grouped[g], x_unfolded[:, g])
        out.append(group_out)
    
    # Stack group outputs
    out = torch.stack(out, dim=1)
    
    # Reshape to final output format
    out = out.view(batch_size, out_channels, out_height, out_width)
    
    # Add bias if provided
    if bias is not None:
        out += bias.view(1, -1, 1, 1)
    
    return out

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
        self.weight, self.bias = self._get_conv_weight_and_bias(
            kernel_size, num_groups, input_channels, output_channels, bias
        )
        self.groups = num_groups
        self.stride = stride
        self.dilation = dilation

    def _get_conv_weight_and_bias(
        self,
        filter_size: Tuple[int, int],
        num_groups: int,
        input_channels: int,
        output_channels: int,
        bias: bool
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert input_channels % num_groups == 0, "input channels must be divisible by groups number"
        assert output_channels % num_groups == 0, "output channels must be divisible by groups number"
        
        in_channels_per_group = input_channels // num_groups
        weight_matrix = torch.randn(output_channels, in_channels_per_group, *filter_size)
        bias_vector = torch.ones(output_channels) if bias else None
        return weight_matrix, bias_vector

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return manual_conv2d(
            x,
            self.weight,
            bias=self.bias,
            stride=self.stride,
            dilation=self.dilation,
            groups=self.groups
        )

class MyFilterStub:
    def __init__(self, filter: torch.Tensor, input_channels: int):
        self.filter = filter
        self.input_channels = input_channels
        # For per-channel filtering, we need to reshape the filter
        self.filter_reshaped = (self.filter[None, None, :, :]
                               .repeat(self.input_channels, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Each input channel is convolved with its own filter
        batch_size, channels, height, width = x.shape
        # Use manual_conv2d with the reshaped filter
        return manual_conv2d(
            x,
            self.filter_reshaped,
            groups=channels
        )