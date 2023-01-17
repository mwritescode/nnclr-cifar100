"""
Implementation from the lightning-bolts library.

References:
    - https://arxiv.org/pdf/1708.03888.pdf
    - https://github.com/pytorch/pytorch/blob/1.6/torch/optim/sgd.py
"""
import math
import torch
from torch.optim.optimizer import Optimizer, required

def configure_parameter_groups(model, cfg):
    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    cls = set()
    for param_name, param in model.named_parameters():
        if 'classifier' in param_name:
            cls.add(param_name)
        elif param_name.endswith('bias'):
            # all biases will not be decayed
            no_decay.add(param_name)
        elif param_name.endswith('weight'):
            if 'bn' in param_name:
                # weights of whitelist modules will be weight decayed
                no_decay.add(param_name)
            else:
                # weights of blacklist modules will NOT be weight decayed
                decay.add(param_name)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay & cls
    union_params = decay | no_decay | cls
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay or cls sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                % (str(param_dict.keys() - union_params), )

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": cfg.WEIGHT_DECAY, "lr": cfg.LR},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0, "lr": cfg.LR},
        {"params": [param_dict[pn] for pn in sorted(list(cls))], "weight_decay": 0.0, "lr": cfg.CLASSIFIER_LR}
    ]
    return optim_groups

class LinearWarmupWithCosineAnnealing:
    def __init__(self, num_warmup_steps: int, num_training_steps: int, last_epoch=-1) -> None:
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.last_epoch = last_epoch
    
    def __call__(self, current_step):
        if current_step < self.num_warmup_steps:
            out =  float(current_step) / float(max(1, self.num_warmup_steps))
        else:
            progress = float(current_step - self.num_warmup_steps) / float(max(1, self.num_training_steps - self.num_warmup_steps))
            out = max(0.0, 0.5 * (1.0 + math.cos(math.pi * 2.0 * progress)))
        return out

class LARS(Optimizer):
    """Extends SGD in PyTorch with LARS scaling from the paper
    `Large batch training of Convolutional Networks <https://arxiv.org/pdf/1708.03888.pdf>`

    .. warning::
        Parameters with weight decay set to 0 will automatically be excluded from
        layer-wise LR scaling. This is to ensure consistency with papers like SimCLR
        and BYOL.
    """

    def __init__(
        self,
        params,
        lr=required,
        momentum=0.0,
        dampening=0.0,
        weight_decay=0.0,
        nesterov=False,
        trust_coefficient=0.001,
        eps=1e-8,
    ):
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            trust_coefficient=trust_coefficient,
            eps=eps,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

        for group in self.param_groups:
            group.setdefault("nesterov", False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # exclude scaling for params with 0 weight decay
        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad
                p_norm = torch.norm(p.data)
                g_norm = torch.norm(p.grad.data)

                # lars scaling + weight decay part
                if weight_decay != 0:
                    if p_norm != 0 and g_norm != 0:
                        lars_lr = p_norm / (g_norm + p_norm * weight_decay + group["eps"])
                        lars_lr *= group["trust_coefficient"]

                        d_p = d_p.add(p, alpha=weight_decay)
                        d_p *= lars_lr

                # sgd part
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-group["lr"])

        return loss