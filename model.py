import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet101

class CSRAModule(nn.Module):
    """Class-Specific Residual Attention module."""
    def __init__(self, in_channels, num_classes, num_heads=1, lam=0.1):
        super(CSRAModule, self).__init__()
        self.num_heads = num_heads
        self.lam = lam
        self.num_classes = num_classes
        
        # Attention branch: Tạo attention map cho mỗi lớp
        self.attention_convs = nn.ModuleList([
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
            for _ in range(num_heads)
        ])
        
        # Classifier branch: Linear layer cho global features
        self.classifier = nn.Linear(in_channels, num_classes)
        
    def forward(self, x):
        # x: [batch_size, in_channels, H, W]
        batch_size = x.size(0)
        
        # Classifier branch: Average pooling + linear
        pooled = F.adaptive_avg_pool2d(x, (1, 1)).view(batch_size, -1)  # [batch_size, in_channels]
        logits_cls = self.classifier(pooled)  # [batch_size, num_classes]
        
        # Attention branch
        logits_att = []
        for conv in self.attention_convs:
            att_map = conv(x)  # [batch_size, num_classes, H, W]
            att_map = F.softmax(att_map.view(batch_size, self.num_classes, -1), dim=2)
            att_map = att_map.view(batch_size, self.num_classes, x.size(2), x.size(3))
            att_features = torch.einsum('bcn,bchw->bc', att_map, x)  # [batch_size, num_classes]
            logits_att.append(att_features)
        
        # Kết hợp các đầu attention
        logits_att = sum(logits_att) / self.num_heads  # [batch_size, num_classes]
        
        # Residual connection với trọng số lam
        logits = logits_cls + self.lam * logits_att
        
        return logits

class MedCSRAModel(nn.Module):
    def __init__(self, num_classes=2468, num_heads=1, lam=0.1):
        super(MedCSRAModel, self).__init__()
        # ResNet-101 backbone
        resnet = resnet101(pretrained=True)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        self.csra = CSRAModule(
            in_channels=2048,  # ResNet-101 layer4 output
            num_classes=num_classes,
            num_heads=num_heads,
            lam=lam
        )
        
        # Fine-tune layer4
        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.backbone[6].parameters():  # layer4
            param.requires_grad = True
    
    def forward(self, pixel_values):
        # pixel_values: [batch_size, 3, 224, 224]
        features = self.backbone(pixel_values)  # [batch_size, 2048, 7, 7]
        logits = self.csra(features)  # [batch_size, num_classes]
        return {"logits_concept": logits}
