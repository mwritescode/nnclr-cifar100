import torch
from torch import nn
from torchvision import models

class LinearEvaluatorModel(nn.Module):
    def __init__(self, num_classes=100) -> None:
        super().__init__()
        resnet18 = models.resnet18()
        resnet18.conv1 = nn.Conv2d(
                    3, 64, kernel_size=3, stride=1, padding=2, bias=False
                )
        resnet18.maxpool = nn.Identity()
        self.backbone = nn.Sequential(*list(resnet18.children())[:-1])
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        hidden_size = resnet18.fc.in_features
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.criterion = nn.CrossEntropyLoss()
    
    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
    
    def forward(self, x, labels=None):
        with torch.no_grad():
            feats = self.backbone(x).squeeze()
        logits = self.classifier(feats)
        out = (logits, )
        if labels is not None:
            loss = self.criterion(logits, labels)
            out += (loss, )

        return out