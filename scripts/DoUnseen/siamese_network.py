import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.detection.backbone_utils


class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.backbone = torchvision.models.vit_b_16()
        self.backbone.heads = nn.Identity()

        self.cls_head = nn.Sequential(nn.Linear(768,768))
        self.act = torch.nn.Hardtanh(-1, 1)
        self.dist = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, img1, img2):
        feat1 = self.backbone(img1)
        feat2 = self.backbone(img2)

        b1 = self.cls_head(feat1)
        b2 = self.cls_head(feat2)

        output = self.dist(b1, b2).unsqueeze(dim=1)
        output = self.act(output)  # hack to fix a bug of numerical stability in pytorch cosine similarity that 1.00000 is slightly bigger than 1.000001 check https://github.com/pytorch/pytorch/issues/78064
        return torch.abs(output)

    def extract_gallery_feats(self, img1):
        feat1 = self.backbone(img1)
        b1 = self.cls_head(feat1)
        return b1

    def extract_query_feats(self, img2):
        feat2 = self.backbone(img2)
        b2 = self.cls_head(feat2)
        return b2

    def dist_measure(self, b1, b2):
        output = self.dist(b1, b2).unsqueeze(dim=1)
        output = self.act(output)  # hack to fix a bug of numerical stability in pytorch cosine similarity that 1.00000 is slightly bigger than 1.000001 check https://github.com/pytorch/pytorch/issues/78064
        return torch.abs(output)
