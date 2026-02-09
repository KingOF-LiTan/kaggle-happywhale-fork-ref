import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class CurricularFace(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super(CurricularFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.threshold = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.register_buffer('t', torch.zeros(1))

    def forward(self, input, label):
        # cos(theta) & phi(theta)
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)

        output = cosine * 1.0  # make copy
        batch_size = input.size(0)
        
        # Target logit
        target_logit = output[torch.arange(batch_size), label]
        
        if self.training:
            with torch.no_grad():
                self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
            
            # Hard sample mining
            mask = cosine > target_logit.unsqueeze(1)
            cosine[mask] = cosine[mask] * (self.t + cosine[mask])
            
        output.scatter_(1, label.unsqueeze(1), phi.unsqueeze(1))
        output *= self.s
        return output
