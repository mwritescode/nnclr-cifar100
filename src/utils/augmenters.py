import random
from PIL import ImageFilter, ImageOps
from torchvision.transforms import RandomSolarize, RandomHorizontalFlip, ColorJitter, InterpolationMode
from torchvision.transforms import ToTensor, Normalize, Compose, RandomGrayscale, RandomApply, RandomResizedCrop

# I think I should use 128 as a threshold for RandomSolarize to obtain the same results as with PIL's imageOps

class ApplyNAugmentationsToImgList:
    def __init__(self, aug_list):
        self.aug_list = aug_list
    
    def __call__(self, img_list):
        out = []
        for aug in self.aug_list:
            out.append([aug(img) for img in img_list]) # [[aug(img) for img in img_list] for aug in self.aug_list]
        return out

class GaussianBlur:
    def __init__(self, sigma=[0.1, 0.2]):
        self.sigma = sigma
    
    def __call__(self, img):
        sigma = random.uniform(*self.sigma)
        return img.filter(ImageFilter.GaussianBlur(radius=sigma))

class Solarize:
    def __call__(self, img):
        return ImageOps.solarize(img)

def get_train_augmentation_pipelines(cfg, stats=None, n_views=2):
    augmentations = []
    random_crop = RandomResizedCrop(
            size=cfg.CROP_SIZE, 
            scale=cfg.RCC.SCALE, 
            interpolation=InterpolationMode.BICUBIC)

    for i in range(n_views):
        augmentations.append([
            random_crop,
            RandomHorizontalFlip(p=cfg.HORIZONTAL_FLIP.PROB[i]),
            RandomApply([ColorJitter(
                brightness=cfg.COLOR_JITTER.BRIGHTNESS[i],
                contrast=cfg.COLOR_JITTER.CONTRAST[i],
                saturation=cfg.COLOR_JITTER.SATURATION[i],
                hue=cfg.COLOR_JITTER.HUE[i]
            )], p=cfg.COLOR_JITTER.PROB[i]),
            RandomGrayscale(p=cfg.GRAYSCALE.PROB[i]),
            RandomApply([GaussianBlur()], p=cfg.GAUSSIAN_BLUR.PROB[i]),
            RandomApply([Solarize()], p=cfg.SOLARIZATION.PROB[i]),
            ToTensor()
        ])
        if stats is not None:
            augmentations[i].append(Normalize(*stats))

    augmentations = [Compose(aug) for aug in augmentations]
    return augmentations

    