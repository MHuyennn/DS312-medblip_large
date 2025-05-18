import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class CSRA(nn.Module):
    def __init__(self, in_features, num_classes, num_heads=1, lam=0.1, dropout=0.5):
        super(CSRA, self).__init__()
        self.num_heads = num_heads
        self.in_features = in_features
        self.head_dim = in_features // num_heads
        assert self.head_dim * num_heads == in_features, "in_features must be divisible by num_heads"
        self.att_fc = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim) for _ in range(num_heads)])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_features, num_classes)
        self.lam = lam

    def forward(self, x):
        batch_size, in_features, height, width = x.size()
        x_flat = x.view(batch_size, in_features, -1)
        x_split = x_flat.view(batch_size, self.num_heads, self.head_dim, height * width)
        att_maps = []
        for i in range(self.num_heads):
            x_head = x_split[:, i, :, :]
            att_map = torch.sigmoid(self.att_fc[i](x_head.permute(0, 2, 1)))
            att_map = att_map.permute(0, 2, 1)
            att_maps.append(att_map)
        att_map = torch.cat(att_maps, dim=1)
        att_features = torch.einsum('bcn,bcn->bc', att_map, x_flat)
        att_features = self.dropout(att_features)
        logits = self.fc(att_features)
        return logits

class MedCSRAModel(nn.Module):
    def __init__(self, num_classes, num_heads=1, lam=0.1, dropout=0.5):
        super(MedCSRAModel, self).__init__()
        self.backbone = models.densenet121(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.features.children()))
        
        # Đóng băng các layer
        for param in self.backbone.parameters():
            param.requires_grad = False
        # Mở block cuối để fine-tune
        for param in self.backbone.features.denseblock4.parameters():
            param.requires_grad = True
        
        in_features = 1024  # DenseNet121 output features
        self.csra = CSRA(in_features=in_features, num_classes=num_classes, num_heads=num_heads, lam=lam, dropout=dropout)
        self.fc = nn.Linear(in_features, num_classes)
        self.lam = nn.Parameter(torch.tensor(lam))
        self.num_classes = num_classes
        self.num_heads = num_heads

    def forward(self, x):
        features = self.backbone(x)
        logits_csra = self.csra(features)
        features_pooled = features.mean(dim=(2, 3))
        logits_global = self.fc(features_pooled)
        logits = (1 - self.lam) * logits_global + self.lam * logits_csra
        return {"logits_concept": logits}
