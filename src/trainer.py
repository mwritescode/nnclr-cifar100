import os
import tqdm
import wandb
import random
import numpy as np
from abc import ABC, abstractmethod

import torch
from torch.optim.lr_scheduler import LambdaLR

from src.config.config import get_cfg_defaults
from src.data.class_mapping import CLASS_MAPPINGS
from src.utils.training.metrics import k_accuracy
from src.utils.training.optim import configure_parameter_groups, LARS, LinearWarmupWithCosineAnnealing

class Trainer(ABC):
    def __init__(self, model, train_loader, val_loader=None, device='cuda', cfg=get_cfg_defaults()):
        self._set_seed(seed=cfg.SYSTEM.SEED)
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.cfg = cfg
        self.eval_interval = 1
    
    @abstractmethod
    def _train_epoch(self, i):
        pass

    @abstractmethod
    def _eval_epoch(self, i):
        pass
    
    def _restore_checkpoint(self, path):
        print('Restoring checkpoint ', path)
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler = checkpoint['scheduler']
        starting_epoch = checkpoint['epoch'] + 1
        wandb_id = checkpoint['wandb_id']
        return starting_epoch, wandb_id
    
    @abstractmethod
    def _save_checkpoint(self, epoch, path, resume=True):
        pass
    
    def _set_seed(self, seed=42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # safe to call even when the GPU is not availabe

        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Set a fixed value for the hash seed
        os.environ["PYTHONHASHSEED"] = str(seed)
        print(f"Random seed set as {seed}")
    
    def fit(self):
        starting_epoch = 0
        if self.cfg.CHECKPOINT.RESTORE:
            starting_epoch, wandb_id = self._restore_checkpoint(path=self.cfg.CHECKPOINT.RESTORE_FROM)
        else:
            wandb_id = wandb.util.generate_id()

        wandb.init(
            # set the wandb project where this run will be logged
            project=self.cfg.LOG.WANDB_PROJECT,
            name=self.cfg.LOG.WANDB_RUN_NAME,
            id=wandb_id,
            resume='allow',
    
            # track hyperparameters and run metadata
            config={
            **self.cfg,
            "architecture": "resnet18",
            "dataset": "CIFAR-100",
            })

        for i in range(starting_epoch, self.num_epochs):
            log_dict = self._train_epoch(i)
            lr_list = self.scheduler.get_last_lr()
            log_dict['lr'] = lr_list[0]
            if len(lr_list) > 1:
                log_dict['classifier_lr'] = lr_list[-1]

            if (i % self.eval_interval) == 0:
                eval_log_dict = self._eval_epoch(i)
                wandb.log({**log_dict, **eval_log_dict}, step=i)
            else:
                wandb.log(log_dict, step=i)

            if (i % self.cfg.CHECKPOINT.INTERVAL) == 0:
                path = os.path.join(self.cfg.CHECKPOINT.SAVE_TO_FOLDER, f'epoch_{i}.pt')
                self._save_checkpoint(epoch=i, path=path, resume=True)
            
            if i == self.num_epochs -1:
                path = os.path.join(self.cfg.CHECKPOINT.SAVE_TO_FOLDER, 'final.pt')
                self._save_checkpoint(epoch=i, path=path, resume=False)
        
        wandb.finish()
        return wandb_id


class TrainerForPretraining(Trainer):
    def __init__(self, model, train_loader, val_loader=None, device='cuda', cfg=get_cfg_defaults()):
        super().__init__(model, train_loader, val_loader, device=device, cfg=cfg)
        self.do_online_eval = cfg.TRAIN.ONLINE_EVAL
        self.log_emb_interval = cfg.LOG.EMB_INTERVAL
        self.do_log_emb = cfg.LOG.EMBEDDINGS
        self.num_epochs = cfg.TRAIN.NUM_EPOCHS + cfg.TRAIN.WARMUP
        self.eval_interval = cfg.TRAIN.EVAL_INTERVAL

        optim_groups = configure_parameter_groups(model, cfg['TRAIN'])
        num_warmup_steps = cfg.TRAIN.WARMUP * len(train_loader)
        num_training_steps = cfg.TRAIN.NUM_EPOCHS * len(train_loader)
        self.optimizer = LARS(optim_groups, momentum=cfg.TRAIN.MOMENTUM)
        self.scheduler = LambdaLR(
            optimizer=self.optimizer, 
            lr_lambda=LinearWarmupWithCosineAnnealing(
                num_warmup_steps=num_warmup_steps, 
                num_training_steps=num_training_steps)
            )
    
    def _train_epoch(self, i):
        epoch_loss = 0.0
        cls_loss, nnclr_loss = 0.0, 0.0
        cls_acc, cls_acc5 = 0.0, 0.0
        current_step = 1
        self.model.train()
        pbar = tqdm.tqdm(self.train_loader, desc=f'Epoch {i}: Training...', total=len(self.train_loader))
        for data in pbar:
            log_dict = {}
            view1 = data['view1'].to(self.device)
            view2 = data['view2'].to(self.device)
            label = data['label'].to(self.device) if self.do_online_eval else None

            self.optimizer.zero_grad(set_to_none=True)
            out = self.model(view1, view2, label)
            if self.do_online_eval:
                loss =  out.loss + out.cls_loss
                cls_loss += out.cls_loss.item()
                nnclr_loss += out.loss.item()
                cls_acc += out.acc1.item()
                cls_acc5 += out.acc5.item()
                log_dict.update({
                    'nnclr_loss': nnclr_loss / current_step, 
                    'cls_loss': cls_loss / current_step,
                    'acc1': cls_acc / current_step,
                    'acc5': cls_acc5 / current_step
                })
            else:
                loss = out.loss
            epoch_loss += loss.item()
            log_dict['loss'] = epoch_loss / current_step
            pbar.set_postfix(log_dict)

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            current_step += 1
        return log_dict
    
    @torch.no_grad()
    def _eval_epoch(self, i):
        log_emb = (i % self.log_emb_interval) == 0 and self.do_log_emb
        cls_loss = 0.0
        cls_acc, cls_acc5 = 0.0, 0.0
        current_step = 1
        self.model.eval()

        if log_emb:
            columns = ['target', 'embedding']
            emb_table = wandb.Table(columns=columns)

        pbar = tqdm.tqdm(self.val_loader, desc=f'Epoch {i}: Eval...', total=len(self.val_loader))
        for data in pbar:
            log_dict = {}
            img = data['img'].to(self.device)
            label = data['label'].to(self.device) if self.do_online_eval else None

            out = self.model(img, labels=label)
            if log_emb:
                for embeds, idx in zip(out.f1.cpu(), label.cpu()):
                    emb_table.add_data(
                        CLASS_MAPPINGS[int(idx.item())],
                        embeds.detach().numpy().tolist()
                    )

            cls_loss += out.cls_loss.item()
            cls_acc += out.acc1.item()
            cls_acc5 += out.acc5.item()
            log_dict.update({
                    'eval_cls_loss': cls_loss / current_step,
                    'eval_acc1': cls_acc / current_step,
                    'eval_acc5': cls_acc5 / current_step
            })
            pbar.set_postfix(log_dict)

            current_step += 1

        if log_emb:
            log_dict[f'embeddings_{i}'] = emb_table
        return log_dict
    
    def _save_checkpoint(self, epoch, path, resume=True):
        if resume:
            # If it's kmclr also save the values in the kmeans.cluster_centers
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler': self.scheduler,
                'wandb_id': wandb.run.id
                }, path)
        else:
            torch.save(self.model.backbone.state_dict(prefix='backbone.'), path)
        
class TrainerForLinearEval(Trainer):
    def __init__(self, model, train_loader, val_loader=None, device='cuda', cfg=get_cfg_defaults(), test_loader=None):
        super().__init__(model, train_loader, val_loader, device, cfg)
        self.num_epochs = cfg.LINEAR_EVAL.EPOCHS + cfg.LINEAR_EVAL.WARMUP
        num_warmup_steps = cfg.LINEAR_EVAL.WARMUP * len(train_loader)
        num_training_steps = cfg.LINEAR_EVAL.EPOCHS * len(train_loader)
        self.optimizer = LARS(model.classifier.parameters(), momentum=cfg.LINEAR_EVAL.MOMENTUM, lr=cfg.LINEAR_EVAL.LR)
        self.scheduler = LambdaLR(
            optimizer=self.optimizer, 
            lr_lambda=LinearWarmupWithCosineAnnealing(
                num_warmup_steps=num_warmup_steps, 
                num_training_steps=num_training_steps)
            )
        self.test_loader = test_loader
    
    def _train_epoch(self, i):
        self.model.train()
        global_loss = 0.0
        global_acc1 = 0.0
        global_acc5 = 0.0
        current_step = 1
        log_dict = {}

        pbar = tqdm.tqdm(self.train_loader, desc=f'Epoch {i}: Training...', total=len(self.train_loader))

        for batch in pbar:
            x = batch['img'].to('cuda')
            y = batch['label'].to('cuda')
            self.optimizer.zero_grad(set_to_none=True)
            logits, loss = self.model(x, y)
            global_loss += loss.item()
            acc1, acc5 = k_accuracy(logits, y, k=5)
            global_acc1 += acc1.item()
            global_acc5 += acc5.item()
            log_dict.update({
                'loss': global_loss / current_step, 
                'acc1': global_acc1 / current_step,
                'acc5': global_acc5 / current_step
            })     
            pbar.set_postfix(log_dict)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            current_step += 1
        return log_dict

    @torch.no_grad()
    def _eval_epoch(self, i, dataset=None):
        self.model.eval()
        global_loss = 0.0
        global_acc1 = 0.0
        global_acc5 = 0.0
        current_step = 1
        log_dict = {}

        if dataset is None:
            dataset = self.val_loader
        pbar = tqdm.tqdm(dataset, desc=f'Epoch {i}: Evaluating...', total=len(dataset))

        for batch in pbar:
            x = batch['img'].to('cuda')
            y = batch['label'].to('cuda')

            logits, loss = self.model(x, y)
            global_loss += loss.item()
            acc1, acc5 = k_accuracy(logits, y, k=5)
            global_acc1 += acc1.item()
            global_acc5 += acc5.item()
            log_dict.update({
                'loss': global_loss / current_step, 
                'acc1': global_acc1 / current_step,
                'acc5': global_acc5 / current_step
            })     
            pbar.set_postfix(log_dict)
            
            current_step += 1
        return log_dict
    
    def _save_checkpoint(self, epoch, path, resume=True):
        if resume:
            # If it's kmclr also save the values in the kmeans.cluster_centers
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler': self.scheduler,
                'wandb_id': wandb.run.id
                }, path)
        else:
            torch.save(self.model.state_dict(), path)
    
    def fit(self):
        wandb_id = super().fit()
        if self.test_loader is not None:
            test_log = self._eval_epoch(0, dataset=self.test_loader)
            api = wandb.Api()
            run = api.run(f"mwritescode/{self.cfg.LOG.WANDB_PROJECT}/{wandb_id}")
            for key in test_log:
                run.summary["test_" + key] = test_log[key]
            run.summary.update()

