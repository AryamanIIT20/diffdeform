import torch
from torch import nn
import numpy as np

class HistLayer(nn.Module):
    """Deep Neural Network Layer for Computing Differentiable Histogram.

    Computes a differentiable histogram
    """

    def __init__(self, in_channels, num_bins=4):
        super(HistLayer, self).__init__()

        # histogram data
        self.in_channels = in_channels
        self.numBins = num_bins
        self.learnable = False
        # self.width = 0.5/(num_bins - 1)
        self.centers = np.linspace(0,1,num_bins)
        self.width = (self.centers[1]-self.centers[0])/2

        # prepare NN layers for histogram computation
        self.bin_centers_conv = nn.Conv2d(
            self.in_channels,
            self.numBins * self.in_channels,
            1,
            groups=self.in_channels,
            bias=True,
        )
        self.bin_centers_conv.weight.data.fill_(1)
        self.bin_centers_conv.weight.requires_grad = False
        self.bin_centers_conv.bias.data = torch.nn.Parameter(
            -torch.tensor(self.centers, dtype=torch.float32)
        )
        self.bin_centers_conv.bias.requires_grad = self.learnable

        self.bin_widths_conv = nn.Conv2d(
            self.numBins * self.in_channels,
            self.numBins * self.in_channels,
            1,
            groups=self.numBins * self.in_channels,
            bias=True,
        )
        self.bin_widths_conv.weight.data.fill_(-1)
        self.bin_widths_conv.weight.requires_grad = False
        self.bin_widths_conv.bias.data.fill_(self.width)
        self.bin_widths_conv.bias.requires_grad = self.learnable
        self.threshold = nn.Threshold(1, 0)
        self.hist_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, input_image, return_2d = False):
        """Computes differentiable histogram.
        Args:
            input_image: input image.
        Returns:
            flattened and un-flattened histogram.
        """
        # |x_i - u_k|
        xx = self.bin_centers_conv(input_image)
        xx = torch.abs(xx)

        # w_k - |x_i - u_k|
        xx = self.bin_widths_conv(xx)

        # 1.01^(w_k - |x_i - u_k|)
        xx = torch.pow(torch.empty_like(xx).fill_(1.01), xx)

        # Φ(1.01^(w_k - |x_i - u_k|), 1, 0)
        xx = self.threshold(xx)
    
        xx_sum = xx.sum([2,3])
        one_d = torch.flatten(xx_sum, 1) 
        if return_2d:
            return one_d/torch.sum(one_d, dim = -1, keepdims = True), torch.flatten(xx, 2)
        else:
            return one_d/torch.sum(one_d, dim = -1, keepdims = True)
