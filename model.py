import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class CSRA(nn.Module):
    def __init__(self, in_features, num_classes, lam=0.1):
        super(CSRA, self).__init__()
        self.fc = nn.Linear(in_features, num_classes)
        self.lam = lam

    def forward(self, x):
        # x: [batch_size, in_features, height, width]
        batch_size, in_features, height, width = x.size()
        
        # Tính attention map
        x_flat = x.view(batch_size, in_features, -1)  # [batch_size, in_features, height*width]
        att_map = torch.sigmoid(self.fc(x_flat.permute(0, 2, 1)))  # [batch_size, height*width, num_classes]
        att_map = att_map.permute(0, 2, 1)  # [batch_size, num_classes, height*width]
        
        # Einsum để tính attention-weighted features
        att_features = torch.einsum('bcn,bchn->bc', att_map, x.view(batch_size, in_features, height*width))  # [batch_size, num_classes]
        
        return att_features

class MedCSRAModel(nn.Module):
    def __init__(self, num_classes=2468, num_heads=1, lam=0.1):
        super(MedCSRAModel, self).__init__()
        # Load ResNet-101 pre-trained
        self.backbone = models.resnet101(pretrained=True)
        
        # Lấy đặc trưng từ layer4 (trước global average pooling)
        self.backbone = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4
        )
        
        # Lấy số lượng đặc trưng từ layer4
        in_features = 2048  # ResNet-101 layer4 output features
        self.csra = CSRA(in_features=in_features, num_classes=num_classes, lam=lam)
        self.fc = nn.Linear(in_features, num_classes)
        
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.lam = lam

    def forward(self, x):
        # x: [batch_size, 3, height, width]
        features = self.backbone(x)  # [batch_size, in_features, height, width]
        
        # CSRA branch
        logits_csra = self.csra(features)  # [batch_size, num_classes]
        
        # Global branch
        features_pooled = features.mean(dim=(2, 3))  # [batch_size, in_features]
        logits_global = self.fc(features_pooled)  # [batch_size, num_classes]
        
        # Kết hợp
        logits = (1 - self.lam) * logits_global + self.lam * logits_csra
        
        return {"logits_concept": logits}
