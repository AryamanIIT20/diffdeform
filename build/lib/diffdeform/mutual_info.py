import torch
from torch import nn
import numpy as np
from .hist_layer import HistLayer
from .rescale import rescale
from .entropy import entropy

def MutualInformation(img1, img2, bins1, bins2):
    img1 = rescale(img1)
    img2 = rescale(img2)
    histlayer1 = HistLayer(in_channels = 1, num_bins = 64)
    histlayer2 = HistLayer(in_channels = 1, num_bins = 64)
    hist1, hist2d1 = histlayer1(img1, return_2d = True)
    hist2, hist2d2 = histlayer2(img2, return_2d = True)
    
    indicator = torch.bmm(hist2d1, hist2d2.transpose(2, 1))
    pxy = indicator/torch.sum(indicator, dim = (1,2), keepdims = True)
    px = torch.sum(pxy, axis=1)  # marginal for x over y
    py = torch.sum(pxy, axis=2)  # marginal for y over x
    mi = entropy(px) + entropy(py) - entropy(pxy)
    return mi


