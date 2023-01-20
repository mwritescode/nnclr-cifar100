import torch
from torch import nn
import torch.nn.functional as F

class QueueSupportSet(nn.Module):
    def __init__(self, embed_size=256, queue_size=10_000) -> None:
        super().__init__()
        self.size = queue_size
        self.embed_size = embed_size
        self.register_buffer("queue", tensor=torch.randn(queue_size, embed_size, dtype=torch.float))
        self.register_buffer("queue_pointer", tensor=torch.zeros(1, dtype=torch.long))
    
    @torch.no_grad()
    def update_queue(self, batch: torch.Tensor):
        batch_size, _ = batch.shape
        pointer = int(self.queue_pointer)

        if pointer + batch_size >= self.size:
            self.queue[pointer:, :] = batch[:self.size - pointer].detach()
            self.queue_pointer[0] = 0
        else:
            self.queue[pointer:pointer + batch_size, :] = batch.detach()
            self.queue_pointer[0] = pointer + batch_size
    
    def forward(self, x, normalized=True):
        queue_l2 = F.normalize(self.queue, p=2, dim=1)
        x_l2 = F.normalize(x, p=2, dim=1)
        similarities = x_l2 @ queue_l2.T
        nn_idx = similarities.argmax(dim=1)
        if normalized:
            out = queue_l2[nn_idx]
        else:
            out = self.queue[nn_idx]
        return out
        