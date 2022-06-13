# encoding: utf-8
# 该实验将卷积操作变成了involuntion算子
import os

import Involution.network_blocks
import yolox.models.network_blocks
from exps.example.yolox_voc import yolox_voc_s
from exps.myexp import tools


class Exp(yolox_voc_s.Exp):
    def __init__(self):
        super(Exp, self).__init__()
        self.num_classes = 10
        self.depth = 0.33
        self.width = 0.50
        self.warmup_epochs = 1

        # ---------- transform config ------------ #
        self.mosaic_prob = 1.0
        self.mixup_prob = 1.0
        self.hsv_prob = 1.0
        self.flip_prob = 0.5

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        self.gpu = True

        tools.patch_class(src_class=yolox.models.network_blocks.BaseConv,your_class=Involution.network_blocks.BaseConv_GPU)
