import tqdm
import wandb
from torch.optim.lr_scheduler import LambdaLR

from src.config.config import get_cfg_defaults
from src.data.class_mapping import CLASS_MAPPINGS
from src.utils.training.optim import configure_parameter_groups, LARS, LinearWarmupWithCosineAnnealing

class Trainer:
    def __init__(self, model, train_loader, val_loader=None, device='cuda', cfg=get_cfg_defaults()):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
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
        
        wandb.init(
            # set the wandb project where this run will be logged
            project=cfg.LOG.WANDB_PROJECT,
            name=cfg.LOG.WANDB_RUN_NAME,
    
            # track hyperparameters and run metadata
            config={
            **cfg,
            "architecture": "resnet18",
            "dataset": "CIFAR-100",
            })
    
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

    def fit(self):
        # TODO: add checkpoints saver
        for i in range(self.num_epochs):
            log_dict = self._train_epoch(i)
            log_dict['lr'] = self.scheduler.get_last_lr()[0]
            log_dict['classifier_lr'] = self.scheduler.get_last_lr()[-1]
            if (i % self.eval_interval) == 0:
                eval_log_dict = self._eval_epoch(i)
                wandb.log({**log_dict, **eval_log_dict})
            else:
                wandb.log(log_dict)
        
        wandb.finish()






