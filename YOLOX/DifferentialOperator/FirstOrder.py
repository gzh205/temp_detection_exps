import torch
import torch.nn.functional as F

filter = {
    'sobel': (
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]],
        [[-1, -2, -1],
         [0, 0, 0],
         [1, 2, 1]]
    ),
    'scharr': (
        [[-3, -10, 3],
         [0, 0, 0],
         [3, 10, 3]],
        [[-3, 0, 3],
         [-10, 0, 10],
         [-3, 0, 3]]
    ),
}


# torch.clip(x,0,1) 将x限定在0-1之间

def conv(input: torch.Tensor, channels: torch.Tensor, weights: (torch.Tensor, torch.Tensor)):
    b = input.shape[0]
    x1 = F.conv2d(input=input, weight=weights[0].repeat(b, channels, 1, 1), bias=False, stride=1, padding=1)
    x2 = F.conv2d(input=input, weight=weights[1].repeat(b, channels, 1, 1), bias=False, stride=1, padding=1)
    return (torch.abs(x1) + torch.abs(x2)) // 2
