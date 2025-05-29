import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import resnet50
from lda import LDA


class L2Norm(nn.Module):
    def forward(self, x):
        return F.normalize(x, p=2, dim=1)


class ResNet(nn.Module):
    def __init__(self, num_classes=1000, lda_args=None, feat_dim=256):
        super(ResNet, self).__init__()
        self.lda_args = lda_args
        self.feat_dim = feat_dim

        # Load pretrained ResNet-50 backbone and remove final FC
        self.backbone, backbone_out_dim = resnet50()

        # Projection head
        self.projection_head = self._make_projector(backbone_out_dim, self.feat_dim)

        # Optional classifier if not using LDA
        self.linear = nn.Linear(backbone_out_dim, num_classes)

        # Optional LDA module
        if self.lda_args:
            self.lda = LDA(num_classes, lda_args['lamb'])

    def _make_projector(self, in_dim, out_dim):
        hidden_dim = 4096
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            L2Norm()  
        )

    def _forward_impl(self, x):
        x = self.backbone(x)
        return x

    def forward(self, x, y=None, epoch=0):
        fea = self._forward_impl(x)

        if self.lda_args:
            fea = self.projection_head(fea)
            xc_mean, sigma_w_inv_b, sigma_w, sigma_b, sigma_t, mu = self.lda(fea, y)
            return xc_mean, sigma_w_inv_b, sigma_w, sigma_b, sigma_t, mu
        else:
            out = self.linear(fea)
            return out


