from SwinTransformer.build import Config
from yolox.models.yolo_pafpn import YOLOPAFPN
import build


class SwinYoloPafpn(YOLOPAFPN):
    def __init__(self, depth=1.0, width=1.0, in_features=("dark3", "dark4", "dark5"), in_channels=[256, 512, 1024],
                 depthwise=False, act="silu", config=None):
        super(SwinYoloPafpn, self).__init__(depth, width, in_features, in_channels, depthwise, act)
        self.backbone = build.build_model(config)
