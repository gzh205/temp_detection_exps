import torch
import torch.nn.functional as F

fliter = {
    'laplace': (
        [[0, 1, 0],
         [1, -4, 1],
         [0, 1, 0]]
    )
}


def conv(input: torch.Tensor, channels: torch.Tensor, weights: (torch.Tensor, torch.Tensor)):
    b = input.shape[0]
    return torch.abs(F.conv2d(input=input, weight=weights[0].repeat(b, channels, 1, 1), bias=False, stride=1, padding=1))
