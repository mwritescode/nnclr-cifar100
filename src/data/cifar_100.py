import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torchvision.transforms import ToTensor, Compose, Normalize, RandomResizedCrop, RandomHorizontalFlip

from src.utils.augmenters import ApplyNAugmentationsToImgList, get_train_augmentation_pipelines

from collections import Counter

MEAN =  (0.5071, 0.4865, 0.4409)
STD = (0.2673, 0.2564, 0.2762)

class CIFAR100(Dataset):
    def __init__(self, split='train', include_label=False, augment_cfg=None, normalize=True, linear_eval=False) -> None:
        super().__init__()
        cols_to_remove = ['coarse_label'] if include_label else ['coarse_label', 'fine_label']
        self.include_label = include_label

        actual_split_in_hub = 'train' if split == 'dev' else split
        self.data = load_dataset('cifar100', split=actual_split_in_hub).remove_columns(cols_to_remove)
        if include_label:
            self.data = self.data.rename_column('fine_label', 'label')

        train_ids, val_ids = self.__get_validation_indexes(num_elem_per_class=50)
        if split == 'train':
            self.data = self.data.select(train_ids)
        elif split == 'dev':
            self.data = self.data.select(val_ids)

        stats = (MEAN, STD) if normalize else None
        self.to_tensor = ToTensor()
        if split == 'train' and not linear_eval:
            aug_list = get_train_augmentation_pipelines(cfg=augment_cfg, stats=stats)
            self.augment = ApplyNAugmentationsToImgList(aug_list=aug_list)
            self.data.set_transform(self.__augment_nviews)
        else:
            if split == 'train':
                self.augment = Compose([
                    RandomResizedCrop(size=32, scale=(0.08, 1.0)),
                    RandomHorizontalFlip(),
                    ToTensor(),
                    Normalize(MEAN, STD)])
            else:
                self.augment = Compose([
                    ToTensor(),
                    Normalize(MEAN, STD)])
            self.data.set_transform(self.__augment_oneview)
    
    def __get_validation_indexes(self, num_elem_per_class=50):
        data_df = self.data.remove_columns('img').to_pandas()
        y = data_df.pop('label')
        X_train, X_val, _, _ = train_test_split(
            data_df.index, y, 
            test_size=num_elem_per_class * 100, 
            stratify=y, random_state=42)
        return X_train, X_val
    
    def __augment_oneview(self, examples):
        examples['original_img'] = [self.to_tensor(img) for img in examples['img']]
        examples['img'] = [self.augment(img) for img in examples['img']]
        if self.include_label:
            examples['label'] = [torch.tensor(label) for label in examples['label']]
        return examples
    
    def __augment_nviews(self, examples):
        examples['view1'], examples['view2'] = self.augment(examples['img'])
        examples['img'] = [self.to_tensor(img) for img in examples['img']]
        if self.include_label:
            examples['label'] = [torch.tensor(label) for label in examples['label']]
        return examples
    
    def __getitem__(self, idx) -> dict:
        out = self.data[idx]
        out['idx'] = torch.tensor(idx)
        return out
    
    def __len__(self) -> int:
        return len(self.data)