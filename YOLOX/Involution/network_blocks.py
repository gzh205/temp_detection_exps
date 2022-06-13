from torch import nn

from involution_cuda import involution as involution_gpu
from involution_naive import involution as involution_cpu
from yolox.models.network_blocks import get_activation


class BaseConv_GPU(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
            self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super(BaseConv_GPU, self).__init__()
        if stride == 1:
            if in_channels == out_channels:
                self.conv = involution_gpu(channels=in_channels, kernel_size=ksize, stride=stride, bias=bias, act=act)
            elif in_channels > out_channels:
                self.conv = nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
                    involution_gpu(channels=out_channels, kernel_size=ksize, stride=stride, bias=bias, act=act)
                )
            elif in_channels < out_channels:
                self.conv = nn.Sequential(
                    involution_gpu(channels=out_channels, kernel_size=ksize, stride=stride, bias=bias, act=act),
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
                )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=(ksize - 1) // 2,
                          groups=groups, bias=bias, ),
                nn.BatchNorm2d(out_channels),
                get_activation(act, inplace=True)
            )

    def forward(self, x):
        return self.conv(x)

    def fuseforward(self, x):
        raise Exception('错误因为没有实现fuseforward')


class BaseConv_CPU(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"):
        super(BaseConv_CPU, self).__init__()
        if stride == 1:
            if in_channels == out_channels:
                self.conv = involution_cpu(channels=in_channels, kernel_size=ksize, stride=stride, bias=bias, act=act)
            elif in_channels > out_channels:
                self.conv = nn.Sequential(
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
                    involution_cpu(channels=out_channels, kernel_size=ksize, stride=stride, bias=bias, act=act)
                )
            elif in_channels < out_channels:
                self.conv = nn.Sequential(
                    involution_cpu(channels=out_channels, kernel_size=ksize, stride=stride, bias=bias, act=act),
                    nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
                )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=(ksize - 1) // 2,
                          groups=groups, bias=bias, ),
                nn.BatchNorm2d(out_channels),
                get_activation(act, inplace=True)
            )

    def forward(self, x):
        return self.conv(x)

    def fuseforward(self, x):
        raise Exception('错误因为没有实现fuseforward')
