import os

from exps.example.yolox_voc import yolox_voc_s


class Exp(yolox_voc_s.Exp):
    def __init__(self):
        super(Exp, self).__init__()
        self.act = 'hswish'
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
