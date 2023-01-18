import torch
from torch import nn
from torchvision import models

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from src.utils.training.loss import NNCLRLoss
from src.utils.training.metrics import k_accuracy
from src.utils.training.modeling_out import NNCLRModelOutput, NNCLRModelOutputWithLinearEval


class KMCLR(nn.Module):
    def __init__(self, embed_size=256, n_clusters=200, projection_hidden_size=2048, prediction_hidden_size=4096, online_eval=True, num_classes=100, reset_interval=1_000) -> None:
        super().__init__()
        self.online_eval = online_eval
        self.reset_interval = reset_interval
        self.n_clusters = n_clusters
        self.step = 1

        resnet18 = models.resnet18()
        resnet18.conv1 = nn.Conv2d(
                    3, 64, kernel_size=3, stride=1, padding=2, bias=False
                )
        resnet18.maxpool = nn.Identity()
        self.backbone = nn.Sequential(*list(resnet18.children())[:-1])
        hidden_size = resnet18.fc.in_features

        self.projection_mlp = nn.Sequential(
            nn.Linear(hidden_size, projection_hidden_size),
            nn.BatchNorm1d(projection_hidden_size),
            nn.ReLU(),
            nn.Linear(projection_hidden_size, projection_hidden_size),
            nn.BatchNorm1d(projection_hidden_size),
            nn.ReLU(),
            nn.Linear(projection_hidden_size, embed_size),
            nn.BatchNorm1d(embed_size)
        )

        self.prediction_mlp = nn.Sequential(
            nn.Linear(embed_size, prediction_hidden_size),
            nn.BatchNorm1d(prediction_hidden_size),
            nn.ReLU(),
            nn.Linear(prediction_hidden_size, embed_size)
        )

        if self.online_eval:
            self.classifier = nn.Linear(hidden_size, num_classes)
            self.cls_criterion = nn.CrossEntropyLoss()
        self.model_output_cls = NNCLRModelOutputWithLinearEval if self.online_eval else NNCLRModelOutput

        self.kmeans = MiniBatchKMeans(n_clusters=n_clusters)
        self.kmeans.partial_fit(np.random.rand(n_clusters, embed_size))
        self.criterion = NNCLRLoss()
    
    def forward(self, x1, x2=None, labels=None):
        # x1 and x1 are two views of the same batch, if both are given we are in the pre-training phase
        # if only x1 is given we are in the online linear evaluation phase for validation set

        f1 = self.backbone(x1).squeeze()
        f2 = proj1 = proj2 = loss = None

        if x2 is not None:
            f2 = self.backbone(x2).squeeze()

            proj1, proj2 = self.projection_mlp(f1), self.projection_mlp(f2)
            pred1, pred2 = self.prediction_mlp(proj1), self.prediction_mlp(proj2)

            tensor_centroids = torch.tensor(self.kmeans.cluster_centers_)

            nn1 = torch.index_select(
                tensor_centroids, 0, 
                torch.tensor(self.kmeans.predict(proj1.detach().cpu().numpy().astype(float)), 
                dtype=torch.long))
            nn2 = torch.index_select(
                tensor_centroids, 0, 
                torch.tensor(self.kmeans.predict(proj2.detach().cpu().numpy().astype(float)), 
                dtype=torch.long))

            loss = self.criterion(preds=(pred1, pred2), neighbors=(nn1.float().to(proj1.device), nn2.float().to(proj2.device)))

            # Only update queue with training batches
            if self.training:
                if (self.step % self.reset_interval) == 0:
                    self.kmeans = MiniBatchKMeans(
                        n_clusters=self.n_clusters, 
                        init=np.random.rand(self.n_clusters, proj1.shape[1]))
                self.kmeans.partial_fit(proj1.detach().cpu().numpy().astype(float))
                self.step += 1
        
        out_dict = {'f1':f1, 'f2':f2,
            'proj1':proj1, 'proj2':proj2,'loss':loss}

        if self.online_eval:
            cls_loss, logits = self.__compute_cls_loss(f1, labels)
            acc1, acc5 = k_accuracy(logits, labels, k=5)

            if f2 is not None:
                cls_loss2, logits2 = self.__compute_cls_loss(f2, labels)
                cls_loss = (cls_loss + cls_loss2) / 2
                acc1_f2, acc5_f2 = k_accuracy(logits2, labels, k=5)
                acc1 = (acc1 + acc1_f2) / 2
                acc5 = (acc5 + acc5_f2) / 2

            out_dict['cls_loss'] = cls_loss
            out_dict['acc1'] = acc1
            out_dict['acc5'] = acc5 
        
        out = self.model_output_cls(**out_dict)

        return out
    
    def __compute_cls_loss(self, features, labels=None):
        logits = self.classifier(features.detach())
        if labels is not None:
            loss = self.cls_criterion(logits, labels)
        return loss, logits
