import torch

def k_accuracy(scores, targets, k=5):
    # Targets should NOT be one-hot encoded

    with torch.no_grad():
        batch_size = targets.shape[0]

        preds = scores.topk(k, dim=1).indices.T
        correct = torch.eq(preds, targets.view(1, -1).expand_as(preds))

        acck = correct.flatten().float().sum()
        acc1 = correct[:1].flatten().float().sum()
        return (acc1 * 100.0) / batch_size, (acck * 100.0) / batch_size
