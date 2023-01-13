import torch
from typing import Optional
from collections import OrderedDict
from dataclasses import dataclass, fields

class ModelOutput(OrderedDict):
    def __post_init__(self):
        out_fields = fields(self)

        if len(out_fields) == 0:
            raise ValueError('The class has no fields')
        
        for field in out_fields:
            val = getattr(self, field.name)
            if val is not None:
                self[field.name] = val
    
    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = {k: v for (k, v) in self.items()}
            return inner_dict[k]
        else:
            return self.to_tuple()[k]


@dataclass(frozen=True)
class NNCLRModelOutput(ModelOutput):
    f1: torch.FloatTensor = None
    f2: Optional[torch.FloatTensor] = None
    proj1: Optional[torch.FloatTensor] = None
    proj2: Optional[torch.FloatTensor] = None
    loss: Optional[torch.FloatTensor] = None

@dataclass(frozen=True)
class NNCLRModelOutputWithLinearEval(NNCLRModelOutput):
    logits: Optional[torch.FloatTensor] = None
    cls_loss: Optional[torch.FloatTensor] = None
    acc1: Optional[torch.FloatTensor] = None
    acc5: Optional[torch.FloatTensor] = None