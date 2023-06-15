import os
import sys

import torch.nn as nn

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
os.chdir(parent_dir)

from backbone.vit import ViT
from head.topdown_heatmap_simple_head import TopdownHeatmapSimpleHead

__all__ = ['ViTPose']


class ViTPose(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super(ViTPose, self).__init__()
        
        backbone_cfg = {k: v for k, v in cfg['backbone'].items() if k != 'type'}
        head_cfg = {k: v for k, v in cfg['keypoint_head'].items() if k != 'type'}
        
        self.backbone = ViT(**backbone_cfg)
        self.keypoint_head = TopdownHeatmapSimpleHead(**head_cfg)
    
    def forward_features(self, x):
        return self.backbone(x)
    
    def forward(self, x):
        return self.keypoint_head(self.backbone(x))