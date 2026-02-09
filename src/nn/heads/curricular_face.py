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
        
        # Ensure constants match input dtype for AMP compatibility
        cos_m = self.cos_m
        sin_m = self.sin_m
        mm = self.mm
        threshold = self.threshold
        
        phi = cosine * cos_m - sine * sin_m
        phi = torch.where(cosine > threshold, phi, cosine - mm)

        if self.training:
            batch_size = input.size(0)
            target_logit = cosine[torch.arange(batch_size), label]
            
            with torch.no_grad():
                self.t = target_logit.mean() * 0.01 + (1 - 0.01) * self.t
            
            # Hard sample mining: boost non-target logits that are "harder" than target
            mask = cosine > target_logit.unsqueeze(1)
            cosine[mask] = cosine[mask] * (self.t.to(input.dtype) + cosine[mask])
            
        # Standard large margin construction
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        output = (one_hot * phi.to(cosine.dtype)) + ((1.0 - one_hot) * cosine)
        output *= self.s
        
        # Return both marginal logits and original cosine for Forwarder compatibility
        return output, cosine
