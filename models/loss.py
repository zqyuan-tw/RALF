import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_dct as dct
from torchvision.models import vgg19

    
class FreqLoss(nn.Module):
    def __init__(self) -> None:
        super(FreqLoss, self).__init__()
        self.l1 = nn.L1Loss()
        
        w = []
        for i in range(256):
            row = []
            for j in range(256):
                if i + j < 1:
                    row.append(1e-4)
                elif i + j < 8:
                    row.append(1e-2)
                elif i + j < 128:
                    row.append(1.)
                elif i + j < 320:
                    row.append(1e1)
                else:
                    row.append(1e2)
            w.append(row)
        self.w = nn.Parameter(torch.tensor(w), requires_grad=False)
        
    def forward(self, x, y):
        
        fx = torch.abs(dct.dct_2d(x)) * self.w
        fy = torch.abs(dct.dct_2d(y)) * self.w
        
        return self.l1(fx, fy)
    
    
class ReconstructionContrastLoss(nn.Module):
    def __init__(self) -> None:
        super(ReconstructionContrastLoss, self).__init__()
        self.l1 = nn.L1Loss()
        
        vgg_feature = vgg19(weights='DEFAULT').features
        vgg_feature.eval()
        for p in vgg_feature.parameters():
            p.requires_grad = False

        self.features = nn.ModuleList([
            vgg_feature[:2],
            vgg_feature[2:4],
            vgg_feature[4:6],
            vgg_feature[6:10],
            vgg_feature[10:14],
        ])

        self.w = [
            1 / 32, 
            1 / 16, 
            1 / 8, 
            1 / 4,
            1
        ]
        
    def forward(self, x, y, idx):
        contrast_loss = 0
        for layer, weight in zip(self.features, self.w):
            x = layer(x)
            y = layer(y)
            contrast_loss += weight * self.l1(x[idx], y[idx]) / self.l1(x[~idx], y[~idx])
        
        return contrast_loss
    
    
class EmbeddingContrastLoss(nn.Module):
    def __init__(self, center=None, r_real=0.05, r_fake=1.5) -> None:
        super(EmbeddingContrastLoss, self).__init__()
        assert r_real > 0 and r_real <= r_fake, "The radius of real sphere should be smaller !"
        self.r_real = r_real
        self.r_fake = r_fake
        
        self.reduce_dim = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.center = nn.Parameter(torch.zeros((128,)), requires_grad=False) if center is None else center
        
        self.real_sum = 0
        self.count = 0
        
    def forward(self, x, idx):
        x = self.reduce_dim(x)
        x = torch.norm(x - self.center, p=2, dim=1)
        emb_loss = torch.clamp(x[idx] - self.r_real, min=0).mean() + torch.clamp(self.r_fake - x[~idx], min=0).mean()
        
        return emb_loss
    
    def accumulate_real(self, x, idx):
        self.real_sum = self.real_sum + torch.sum(self.reduce_dim(x), dim=0)
        self.count = self.count + idx.sum().item()
    
    def update_center(self):
        assert self.count > 0, "No real sample for an update."
        new_center = self.real_sum / self.count
        with torch.no_grad():
            self.center.data.copy_(new_center)
        self.real_sum = 0
        self.count = 0

