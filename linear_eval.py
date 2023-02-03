import torch
import torchinfo
from argparse import ArgumentParser
from torch.utils.data import DataLoader

from src.trainer import TrainerForLinearEval
from src.models.evaluator import LinearEvaluatorModel
from src.data.cifar_100 import CIFAR100
from src.config.config import get_cfg_defaults


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('config_path', help='Path of the model\'s configuration file')
    args = args.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_path)
    
    train_data = CIFAR100('train', linear_eval=True, include_label=True)
    val_data = CIFAR100('dev', linear_eval=True, include_label=True)
    test_data = CIFAR100('test', linear_eval=True, include_label=True)

    train_loader = DataLoader(
        train_data, batch_size=cfg.LINEAR_EVAL.BATCH_SIZE, shuffle=True,
        pin_memory=True, num_workers=cfg.SYSTEM.NUM_WORKERS)
    val_loader = DataLoader(
        val_data, batch_size=cfg.LINEAR_EVAL.BATCH_SIZE, 
        pin_memory=True, num_workers=cfg.SYSTEM.NUM_WORKERS)
    test_loader = DataLoader(
        test_data, batch_size=cfg.LINEAR_EVAL.BATCH_SIZE, 
        pin_memory=True, num_workers=cfg.SYSTEM.NUM_WORKERS)
    
    model = LinearEvaluatorModel()
    checkpoint = torch.load(cfg.LINEAR_EVAL.CHECKPOINT)
    model.load_state_dict(checkpoint, strict=False)
    torchinfo.summary(model)

    trainer = TrainerForLinearEval(model, train_loader, val_loader, test_loader=test_loader, device=cfg.SYSTEM.DEVICE, cfg=cfg)
    trainer.fit()