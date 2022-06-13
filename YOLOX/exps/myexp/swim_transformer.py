from SwinTransformer.build import Config
from exps.example.yolox_voc import yolox_voc_s
from torch import nn


class Exp(yolox_voc_s.Exp):
    def __init__(self):
        super(Exp, self).__init__()

    def get_model(self):
        from yolox.models import YOLOX, YOLOXHead
        from SwinTransformer.yolo_pafpn import YOLOPAFPN
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            config = Config().set_img_size(self.img_size)
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, act=self.act, config=config)
            head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, act=self.act)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        self.model.train()
        return self.model
