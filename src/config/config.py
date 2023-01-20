from yacs.config import CfgNode as CN

_C = CN()

# System configuration parameters

_C.SYSTEM = CN()
_C.SYSTEM.NUM_WORKERS = 1
_C.SYSTEM.DEVICE = 'cuda'
_C.SYSTEM.SEED = 42

# Training configuration parameters

_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 64
_C.TRAIN.NUM_EPOCHS = 500
_C.TRAIN.WARMUP = 10
_C.TRAIN.LR = 0.2
_C.TRAIN.WEIGHT_DECAY = 1e-6
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.ONLINE_EVAL = True
_C.TRAIN.CLASSIFIER_LR = 0.01
_C.TRAIN.EVAL_INTERVAL = 1
_C.TRAIN.RESET_CENTROIDS_INTERVAL = 3


_C.MODEL = CN()
_C.MODEL.NAME = 'nnclr' # or 'kmclr'
_C.MODEL.EMBED_SIZE = 256
_C.MODEL.PROJ_HIDDEN_SIZE = 2048
_C.MODEL.PRED_HIDDEN_SIZE = 4096
_C.MODEL.NUM_CLASSES = 100
_C.MODEL.NUM_CENTROIDS = 400
_C.MODEL.QUEUE_SIZE = 80_000

# Wandb logging options

_C.LOG = CN()
_C.LOG.WANDB_PROJECT = 'nnclr-cifar100'
_C.LOG.WANDB_RUN_NAME = 'nnclr-pretrain'
_C.LOG.EMBEDDINGS = True
_C.LOG.EMB_INTERVAL = 50

# Checkpointing options during training

_C.CHECKPOINT = CN()
_C.CHECKPOINT.RESTORE = False
_C.CHECKPOINT.RESTORE_FROM = 'checkpoints/model_epoch_5.pt'
_C.CHECKPOINT.SAVE_TO_FOLDER = 'checkpoints'
_C.CHECKPOINT.INTERVAL = 100

# Options for the two augmented views during pre-training

_C.AUGMENT = CN()
_C.AUGMENT.CROP_SIZE = 32

_C.AUGMENT.RCC = CN()
_C.AUGMENT.RCC.SCALE = [0.08, 1.0]

_C.AUGMENT.COLOR_JITTER = CN()
_C.AUGMENT.COLOR_JITTER.PROB = [0.8, 0.8]
_C.AUGMENT.COLOR_JITTER.BRIGHTNESS = [0.4, 0.4]
_C.AUGMENT.COLOR_JITTER.CONTRAST = [0.4,  0.4]
_C.AUGMENT.COLOR_JITTER.SATURATION = [0.2, 0.2]
_C.AUGMENT.COLOR_JITTER.HUE = [0.1, 0.1]

_C.AUGMENT.GRAYSCALE = CN()
_C.AUGMENT.GRAYSCALE.PROB = [0.2, 0.2]

_C.AUGMENT.GAUSSIAN_BLUR = CN()
_C.AUGMENT.GAUSSIAN_BLUR.PROB = [1.0, 0.1]

_C.AUGMENT.SOLARIZATION = CN()
_C.AUGMENT.SOLARIZATION.PROB = [0.0, 0.2]

_C.AUGMENT.HORIZONTAL_FLIP = CN()
_C.AUGMENT.HORIZONTAL_FLIP.PROB = [0.5, 0.5]

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()

def save_cfg_default():
    """
    Save in a YAML file the default version of the configuration file, 
    in order to provide a template to be modified.
    """
    with open('src/config/experiments/default.yaml', 'w') as f:
        f.write(_C.dump())
        f.flush()
        f.close()

if __name__ == '__main__':
    save_cfg_default()