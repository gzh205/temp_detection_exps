# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
from torch import nn

from .swin_transformer import SwinTransformer
from .swin_transformer_v2 import SwinTransformerV2
from .swin_mlp import SwinMLP


class Config():
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0]):
        self.IMG_SIZE = img_size
        self.PATCH_SIZE = patch_size
        self.IN_CHANS = in_chans
        self.NUM_CLASSES = num_classes
        self.EMBED_DIM = embed_dim
        self.DEPTHS = depths
        self.NUM_HEADS = num_heads
        self.WINDOW_SIZE = window_size
        self.MLP_RATIO = mlp_ratio
        self.QKV_BIAS = qkv_bias
        self.QK_SCALE = qk_scale
        self.DROP_RATE = drop_rate
        self.ATTN_DROP_RATE = attn_drop_rate
        self.DROP_PATH_RATE = drop_path_rate
        self.NORM_LAYER = norm_layer
        self.APE = ape
        self.PATCH_NORM = patch_norm
        self.USE_CHECKPOINT = use_checkpoint
        self.PRETRAINED_WINDOW_SIZES = pretrained_window_sizes

    def set_img_size(self, img_size):
        self.IMG_SIZE = img_size
        return self

    def set_patch_size(self, patch_size):
        self.PATCH_SIZE = patch_size
        return self

    def set_in_chans(self, in_chans):
        self.IN_CHANS = in_chans
        return self

    def set_num_classes(self, num_classes):
        self.NUM_CLASSES = num_classes
        return self

    def set_embed_dim(self, embed_dim):
        self.EMBED_DIM = embed_dim
        return self

    def set_depths(self, depths):
        self.DEPTHS = depths
        return self

    def set_num_heads(self, num_heads):
        self.NUM_HEADS = num_heads
        return self

    def set_window_size(self, window_size):
        self.WINDOW_SIZE = window_size
        return self

    def set_mlp_ratio(self, mlp_ratio):
        self.MLP_RATIO = mlp_ratio
        return self

    def set_qkv_bias(self, qkv_bias):
        self.QKV_BIAS = qkv_bias
        return self

    def set_qk_scale(self, qk_scale):
        self.QK_SCALE = qk_scale
        return self

    def set_drop_rate(self, drop_rate):
        self.DROP_RATE = drop_rate
        return self

    def set_attn_drop_rate(self, attn_drop_rate):
        self.ATTN_DROP_RATE = attn_drop_rate
        return self

    def set_drop_path_rate(self, drop_path_rate):
        self.DROP_PATH_RATE = drop_path_rate
        return self

    def set_norm_layer(self, norm_layer):
        self.NORM_LAYER = norm_layer
        return self

    def set_ape(self, ape):
        self.APE = ape
        return self

    def set_patch_norm(self, patch_norm):
        self.PATCH_NORM = patch_norm
        return self

    def set_use_checkpoint(self, use_checkpoint):
        self.USE_CHECKPOINT = use_checkpoint
        return self

    def set_pretrained_window_sizes(self, pretrained_window_sizes):
        self.PRETRAINED_WINDOW_SIZES = pretrained_window_sizes
        return self


def build_model(config: Config):
    model_type = config.MODEL.TYPE
    if model_type == 'swin':
        model = SwinTransformer(img_size=config.IMG_SIZE,
                                patch_size=config.PATCH_SIZE,
                                in_chans=config.SWIN.IN_CHANS,
                                num_classes=config.NUM_CLASSES,
                                embed_dim=config.SWIN.EMBED_DIM,
                                depths=config.DEPTHS,
                                num_heads=config.SWIN.NUM_HEADS,
                                window_size=config.WINDOW_SIZE,
                                mlp_ratio=config.MLP_RATIO,
                                qkv_bias=config.QKV_BIAS,
                                qk_scale=config.QK_SCALE,
                                drop_rate=config.DROP_RATE,
                                drop_path_rate=config.DROP_PATH_RATE,
                                ape=config.APE,
                                patch_norm=config.PATCH_NORM,
                                use_checkpoint=config.USE_CHECKPOINT)
    elif model_type == 'swinv2':
        model = SwinTransformerV2(img_size=config.IMG_SIZE,
                                  patch_size=config.PATCH_SIZE,
                                  in_chans=config.IN_CHANS,
                                  num_classes=config.NUM_CLASSES,
                                  embed_dim=config.EMBED_DIM,
                                  depths=config.DEPTHS,
                                  num_heads=config.NUM_HEADS,
                                  window_size=config.WINDOW_SIZE,
                                  mlp_ratio=config.MLP_RATIO,
                                  qkv_bias=config.QKV_BIAS,
                                  drop_rate=config.DROP_RATE,
                                  drop_path_rate=config.DROP_PATH_RATE,
                                  ape=config.APE,
                                  patch_norm=config.PATCH_NORM,
                                  use_checkpoint=config.USE_CHECKPOINT,
                                  pretrained_window_sizes=config.PRETRAINED_WINDOW_SIZES)
    elif model_type == 'swin_mlp':
        model = SwinMLP(img_size=config.IMG_SIZE,
                        patch_size=config.PATCH_SIZE,
                        in_chans=config.IN_CHANS,
                        num_classes=config.NUM_CLASSES,
                        embed_dim=config.EMBED_DIM,
                        depths=config.DEPTHS,
                        num_heads=config.NUM_HEADS,
                        window_size=config.WINDOW_SIZE,
                        mlp_ratio=config.MLP_RATIO,
                        drop_rate=config.DROP_RATE,
                        drop_path_rate=config.DROP_PATH_RATE,
                        ape=config.APE,
                        patch_norm=config.PATCH_NORM,
                        use_checkpoint=config.USE_CHECKPOINT)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
