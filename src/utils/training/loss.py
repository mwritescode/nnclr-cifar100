import torch
from torch import nn
import torch.nn.functional as F

class NNCLRLoss(nn.Module):
    def __init__(self, reduction='mean', temperature=0.1) -> None:
        super().__init__()
        self.reduction = reduction
        self.temperature = temperature
    
    def __compute_loss(self, pred, nn):
        batch_size, _ = pred.shape
        labels = torch.arange(batch_size).to(pred.device)

        nn = F.normalize(nn, p=2, dim=1)
        pred = F.normalize(pred, p=2, dim=1)

        logits = (nn @ pred.T) / self.temperature

        return F.cross_entropy(logits, labels, reduction=self.reduction)
    
    def forward(self, preds, neighbors):
        pred1, pred2 = preds
        nn1, nn2 = neighbors
        
        loss = self.__compute_loss(pred1, nn2) * 0.5 +\
            + self.__compute_loss(pred2, nn1) * 0.5
        
        return loss
