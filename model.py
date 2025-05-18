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
        # Load DenseNet121 với weights IMAGENET1K_V1
        self.backbone = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        
        # Lấy các layer từ DenseNet121
        backbone_layers = list(self.backbone.children())
        self.backbone = nn.Sequential(*backbone_layers[:-1])  # Loại bỏ lớp fully connected cuối (classifier)
        
        # Đóng băng tất cả các layer
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Mở denseblock4 để fine-tune (denseblock4 là layer thứ 6 trong backbone_layers)
        for param in backbone_layers[6].parameters():  # denseblock4 là layer 6
            param.requires_grad = True
        
        in_features = 1024  # Output features của DenseNet121 trước classifier
        self.csra = CSRA(in_features=in_features, num_classes=num_classes, num_heads=num_heads, lam=lam, dropout=dropout)
        self.fc = nn.Linear(in_features, num_classes)
        self.lam = nn.Parameter(torch.tensor(lam))
        self.num_classes = num_classes
        self.num_heads = num_heads

    def forward(self, x):
        features = self.backbone(x)  # [batch_size, 1024, 7, 7]
        # Global average pooling để giảm kích thước không gian
        features = F.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)  # [batch_size, 1024]
        logits_csra = self.csra(features.unsqueeze(-1).unsqueeze(-1))  # Thêm chiều không gian cho CSRA
        logits_global = self.fc(features)
        logits = (1 - self.lam) * logits_global + self.lam * logits_csra
        return {"logits_concept": logits}
