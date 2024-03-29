import torchinfo
from argparse import ArgumentParser
from torch.utils.data import DataLoader

from src.trainer import TrainerForPretraining
from src.models.nnclr import NNCLR
from src.models.kmclr import KMCLR
from src.data.cifar_100 import CIFAR100
from src.config.config import get_cfg_defaults

def get_model_from_cfg(cfg, reset_interval):
    shared_params = {
        "embed_size": cfg.MODEL.EMBED_SIZE,
        "projection_hidden_size": cfg.MODEL.PROJ_HIDDEN_SIZE, 
        "prediction_hidden_size": cfg.MODEL.PRED_HIDDEN_SIZE, 
        "online_eval": cfg.TRAIN.ONLINE_EVAL, 
        "num_classes": cfg.MODEL.NUM_CLASSES
    }
    if cfg.MODEL.NAME == 'nnclr':
        model = NNCLR(queue_size=cfg.MODEL.QUEUE_SIZE, **shared_params)
    else:
        model = KMCLR(n_clusters=cfg.MODEL.NUM_CENTROIDS, reset_interval=reset_interval, **shared_params)
    return model


if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument('config_path', help='Path of the model\'s configuration file')
    args = args.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config_path)
    
    train_data = CIFAR100('train', augment_cfg=cfg['AUGMENT'], include_label=True)
    val_data = CIFAR100('dev', augment_cfg=cfg['AUGMENT'], include_label=True)

    train_loader = DataLoader(
        train_data, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True,
        pin_memory=True, num_workers=cfg.SYSTEM.NUM_WORKERS)
    val_loader = DataLoader(
        val_data, batch_size=cfg.TRAIN.BATCH_SIZE, 
        pin_memory=True, num_workers=cfg.SYSTEM.NUM_WORKERS)
    
    model = get_model_from_cfg(cfg, reset_interval=cfg.TRAIN.RESET_CENTROIDS_INTERVAL * len(train_loader))
    torchinfo.summary(model)

    trainer = TrainerForPretraining(model, train_loader, val_loader, device=cfg.SYSTEM.DEVICE, cfg=cfg)
    trainer.fit()