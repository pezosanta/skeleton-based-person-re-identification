import yaml
import random
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from models.bnlstm import BNLSTM
from models.siamese import ModelMode, SiameseCrossentropy
from datasets import DatasetType, TrainingMode, NegativePairLabel, SiameseDataset
from utils import create_cm_figure




class SiameseModel(LightningModule):
    def __init__(self):
        super(SiameseModel, self).__init__()

        self.device_type = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # For reproducibility purposes
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        self.now = datetime.now().strftime('%Y.%m.%d-%H:%M:%S')

        self.configs = yaml.safe_load(open('trainers/siamese_trainer_config.yml').read())
            
        self.train_dataset = SiameseDataset(
            dataset_type=DatasetType.TRAINING,
            window_size=self.configs['dataset_window_size'],
            shift_size=self.configs['dataset_shift_size'],
            num_exclude=self.configs['dataset_num_exclude'],
            mode=TrainingMode.SIAMESE,
            need_pair_dataset=True,
            negative_pair_label=NegativePairLabel.CROSSENTROPY,
            use_tree_structure=self.configs['dataset_use_tree_structure'],
            use_scaled_dataset=self.configs['dataset_use_scaled']
        )
        
        self.val_dataset = SiameseDataset(
            dataset_type=DatasetType.VALIDATION,
            window_size=self.configs['dataset_window_size'],
            shift_size=self.configs['dataset_shift_size'],
            num_exclude=self.configs['dataset_num_exclude'],
            mode=TrainingMode.SIAMESE,
            need_pair_dataset=True,
            negative_pair_label=NegativePairLabel.CROSSENTROPY,
            use_tree_structure=self.configs['dataset_use_tree_structure'],
            use_scaled_dataset=self.configs['dataset_use_scaled']
        )

        core_model = BNLSTM(
                input_size=self.train_dataset.base_dataset[0].shape[2],
                lstm_output_size=self.train_dataset.windows_num.shape[0],
                hidden_size=self.configs['bnlstm_hidden_size'],
                n_layers=self.configs['bnlstm_n_layers'],
                dropout=self.configs['bnlstm_dropout']
        )

        self.model = SiameseCrossentropy(
            model=core_model,
            model_output_size=self.train_dataset.windows_num.shape[0],
            latent_size=self.configs['siamese_latent_size'],
            out_fc_size=self.configs['siamese_out_fc_size'],
            latent_fc_dropout=self.configs['siamese_latent_dropout']
        )

        if self.configs['aux_criterion_use']:
            self.aux_criterion_weight = self.configs['aux_criterion_weight']
            self.aux_criterion = nn.CosineEmbeddingLoss(margin=self.configs['aux_criterion_margin'])
        self.criterion = nn.BCELoss()

        self.optimizer = optim.Adam(self.parameters(), lr=self.configs['optimizer_base_lr'], weight_decay=self.configs['optimizer_weight_decay'])
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer, milestones=self.configs['scheduler_milestones'], gamma=self.configs['scheduler_gamma'])



    def forward(self, x1, x2=None, mode=ModelMode.TRAINING):
        out = self.model(x1=x1, x2=x2, mode=mode)

        return out

  

    def configure_optimizers(self):
        return [self.optimizer], [self.lr_scheduler]


    
    def on_train_start(self):
        if self.configs['aux_criterion_use'] == True:
            aux_loss_log =  f"Auxiliary Loss type: {self.aux_criterion.__class__.__name__}\n\n" \
                            f"Auxiliary Loss margin: {self.configs['aux_criterion_margin']}\n\n" \
                            f"Auxiliary Loss weight: {self.configs['aux_criterion_weight']}"
        else:
            aux_loss_log = f"Auxiliary Loss type: None"

        self.logger.experiment.add_text(tag="General Information",
                text_string=
                    f"Start time: {self.now}\n\n" \

                    f"Number of epochs: {self.configs['train_num_epochs']}\n\n" \
                    f"Batch size: {self.configs['train_batch_size']}\n\n" \

                    f"Dataset window size: {self.configs['dataset_window_size']}\n\n" \
                    f"Dataset shift size: {self.configs['dataset_shift_size']}\n\n" \
                    f"Dataset number of excluded person: {self.configs['dataset_num_exclude']}\n\n" \
                    f"Dataset using scaled dataset: {self.configs['dataset_use_scaled']}\n\n" \
                    f"Dataset using tree structure: {self.configs['dataset_use_tree_structure']}\n\n" \

                    f"Model: {self.model.__class__.__name__}\n\n" \
                    f"Model latent dropout: {self.configs['siamese_latent_dropout']}\n\n" \
                    f"Model number of trainable parameters: {self.model.count_parameters()}\n\n" \

                    f"BNLSTM hidden size: {self.configs['bnlstm_hidden_size']}\n\n" \
                    f"BNLSTM number of layers: {self.configs['bnlstm_n_layers']}\n\n" \
                    f"BNLSTM dropout value: {self.configs['bnlstm_dropout']}\n\n" \
                    
                    f"Loss type: {self.criterion.__class__.__name__}\n\n" \
                    f"{aux_loss_log}\n\n" \

                    f"Optimizer type: {self.optimizer.__class__.__name__}\n\n" \
                    f"Optimizer base learning rate: {self.configs['optimizer_base_lr']}\n\n" \
                    f"Optimizer weight decay: {self.configs['optimizer_weight_decay']}\n\n" \

                    f"Scheduler type: {self.lr_scheduler.__class__.__name__}\n\n" \
                    f"Scheduler milestones: {self.configs['scheduler_milestones']}\n\n" \
                    f"Scheduler gamma: {self.configs['scheduler_gamma']}\n\n",

                global_step=0) 

    

    def training_step(self, batch, batch_idx):
        x1, x2, y = batch[0].to(device=self.device_type), batch[1].to(device=self.device_type), batch[2].to(device=self.device_type)  
        
        if self.current_epoch == 0 and batch_idx == 0:
            self.logger.experiment.add_graph(model=self.model, input_to_model=(x1, x2), verbose=False)  

        y = torch.unsqueeze(y, dim=1)
        y_hat, x1, x2 = self.forward(x1, x2)

        loss = self.criterion(y_hat, y)

        if self.configs['aux_criterion_use']:
            # CosineEmbeddingLoss needs label=-1.0 instead of 0.0 for negative pairs
            y_aux = y.clone()
            y_aux[y_aux==0.0] = -1.0   
                 
            aux_loss = self.aux_criterion(x1, x2, y_aux)

            loss = loss + self.aux_criterion_weight*aux_loss

        y_pred_50 = (y_hat > 0.5).type(torch.float32)
        y_pred_80 = (y_hat > 0.8).type(torch.float32)
        y_pred_85 = (y_hat > 0.85).type(torch.float32)
        y_pred_90 = (y_hat > 0.9).type(torch.float32)
        y_pred_95 = (y_hat > 0.95).type(torch.float32)

        acc_50 = accuracy_score(y.cpu().view(-1), y_pred_50.cpu().view(-1))

        # Logging to TensorBoard 
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=False)
        self.log('train_acc_50', acc_50, on_step=True, on_epoch=True, prog_bar=True, logger=False)

        return {'loss': loss, 'y_pred_50': y_pred_50, 'y_pred_80': y_pred_80, 'y_pred_85': y_pred_85, 'y_pred_90': y_pred_90, 'y_pred_95': y_pred_95, 'y': y}



    def training_epoch_end(self, outputs) -> None:
        epoch_loss = torch.stack([out['loss'] for out in outputs]).mean()
        all_y_pred_50 = torch.stack([out['y_pred_50'] for out in outputs]).view(-1)
        all_y_pred_80 = torch.stack([out['y_pred_80'] for out in outputs]).view(-1)
        all_y_pred_85 = torch.stack([out['y_pred_85'] for out in outputs]).view(-1)
        all_y_pred_90 = torch.stack([out['y_pred_90'] for out in outputs]).view(-1)
        all_y_pred_95 = torch.stack([out['y_pred_95'] for out in outputs]).view(-1)
        all_y = torch.stack([out['y'] for out in outputs]).view(-1)

        epoch_acc_50 = accuracy_score(all_y.cpu(), all_y_pred_50.cpu())
        epoch_acc_80 = accuracy_score(all_y.cpu(), all_y_pred_80.cpu())
        epoch_acc_85 = accuracy_score(all_y.cpu(), all_y_pred_85.cpu())
        epoch_acc_90 = accuracy_score(all_y.cpu(), all_y_pred_90.cpu())
        epoch_acc_95 = accuracy_score(all_y.cpu(), all_y_pred_95.cpu())
        
        # Logging metrics to Tensorboard
        self.logger.experiment.add_scalar(tag='Training Loss', scalar_value=(epoch_loss), global_step=self.current_epoch)
        self.logger.experiment.add_scalar(tag='Training Accuracy/Training Accuracy @50', scalar_value=(epoch_acc_50), global_step=self.current_epoch)
        self.logger.experiment.add_scalar(tag='Training Accuracy/Training Accuracy @80', scalar_value=(epoch_acc_80), global_step=self.current_epoch)
        self.logger.experiment.add_scalar(tag='Training Accuracy/Training Accuracy @85', scalar_value=(epoch_acc_85), global_step=self.current_epoch)
        self.logger.experiment.add_scalar(tag='Training Accuracy/Training Accuracy @90', scalar_value=(epoch_acc_90), global_step=self.current_epoch)
        self.logger.experiment.add_scalar(tag='Training Accuracy/Training Accuracy @95', scalar_value=(epoch_acc_95), global_step=self.current_epoch)
      


    def validation_step(self, batch, batch_idx):
        x1, x2, y = batch[0].to(device=self.device_type), batch[1].to(device=self.device_type), batch[2].to(device=self.device_type)  
        
        y = torch.unsqueeze(y, dim=1)
        y_hat, x1, x2 = self.forward(x1, x2)

        loss = self.criterion(y_hat, y)

        if self.configs['aux_criterion_use']:
            # CosineEmbeddingLoss needs label=-1.0 instead of 0.0 for negative pairs
            y_aux = y.clone()
            y_aux[y_aux==0.0] = -1.0        
            aux_loss = self.aux_criterion(x1, x2, y_aux)

            loss = loss + self.aux_criterion_weight*aux_loss

        y_pred_50 = (y_hat > 0.5).type(torch.float32)
        y_pred_80 = (y_hat > 0.8).type(torch.float32)
        y_pred_85 = (y_hat > 0.85).type(torch.float32)
        y_pred_90 = (y_hat > 0.9).type(torch.float32)
        y_pred_95 = (y_hat > 0.95).type(torch.float32)

        acc_50 = accuracy_score(y.cpu().view(-1), y_pred_50.cpu().view(-1))
        

        # Logging to TensorBoard 
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=False)
        self.log('val_acc_50', acc_50, on_step=True, on_epoch=True, prog_bar=True, logger=False)

        return {'loss': loss, 'y_pred_50': y_pred_50, 'y_pred_80': y_pred_80, 'y_pred_85': y_pred_85, 'y_pred_90': y_pred_90, 'y_pred_95': y_pred_95, 'y': y}



    def validation_epoch_end(self, outputs) -> None:
        epoch_loss = torch.stack([out['loss'] for out in outputs]).mean()
        all_y_pred_50 = torch.stack([out['y_pred_50'] for out in outputs]).view(-1)
        all_y_pred_80 = torch.stack([out['y_pred_80'] for out in outputs]).view(-1)
        all_y_pred_85 = torch.stack([out['y_pred_85'] for out in outputs]).view(-1)
        all_y_pred_90 = torch.stack([out['y_pred_90'] for out in outputs]).view(-1)
        all_y_pred_95 = torch.stack([out['y_pred_95'] for out in outputs]).view(-1)
        all_y = torch.stack([out['y'] for out in outputs]).view(-1)

        epoch_acc_50 = accuracy_score(all_y.cpu(), all_y_pred_50.cpu())
        epoch_acc_80 = accuracy_score(all_y.cpu(), all_y_pred_80.cpu())
        epoch_acc_85 = accuracy_score(all_y.cpu(), all_y_pred_85.cpu())
        epoch_acc_90 = accuracy_score(all_y.cpu(), all_y_pred_90.cpu())
        epoch_acc_95 = accuracy_score(all_y.cpu(), all_y_pred_95.cpu())
        
        # Logging metrics to Tensorboard
        self.logger.experiment.add_scalar(tag='Validation Loss', scalar_value=(epoch_loss), global_step=self.current_epoch)
        self.logger.experiment.add_scalar(tag='Validation Accuracy/Validation Accuracy @50', scalar_value=(epoch_acc_50), global_step=self.current_epoch)
        self.logger.experiment.add_scalar(tag='Validation Accuracy/Validation Accuracy @80', scalar_value=(epoch_acc_80), global_step=self.current_epoch)
        self.logger.experiment.add_scalar(tag='Validation Accuracy/Validation Accuracy @85', scalar_value=(epoch_acc_85), global_step=self.current_epoch)
        self.logger.experiment.add_scalar(tag='Validation Accuracy/Validation Accuracy @90', scalar_value=(epoch_acc_90), global_step=self.current_epoch)
        self.logger.experiment.add_scalar(tag='Validation Accuracy/Validation Accuracy @95', scalar_value=(epoch_acc_95), global_step=self.current_epoch)

  

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.configs['train_batch_size'], shuffle=True, drop_last=True, num_workers=0)



    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.configs['train_batch_size'], shuffle=False, drop_last=True, num_workers=0) 



    def initialize_logger(self):
        model_name = f"{self.model.__class__.__name__}" if self.model.__class__.__name__ != "LogisticRegression" else "LOGREG"
        loss_name = f"{self.criterion.__class__.__name__}" if self.criterion.__class__.__name__ != "BCEWithLogitsLoss" else "BCEWLL"

        if self.configs['aux_criterion_use'] == True:
            aux_loss_name = f"Cos" if self.aux_criterion.__class__.__name__ == "CosineEmbeddingLoss" else None
            aux_loss_log =  f"auxLoss_{aux_loss_name}-" \
                            f"auxLossM_{self.configs['aux_criterion_margin']}-" \
                            f"auxLossW_{self.configs['aux_criterion_weight']}"
        else:
            aux_loss_log = "auxLoss_None"

        save_dir = 'tensorboard_logs'
        self.logger_subdir = 'siamese'
        self.logger_run_name =  f"{self.now}-" \
                                f"epochs_{self.configs['train_num_epochs']}-" \
                                f"bs_{self.configs['train_batch_size']}-" \
                                f"dsWS_{self.configs['dataset_window_size']}-" \
                                f"dsSS_{self.configs['dataset_shift_size']}-" \
                                f"dsNE_{self.configs['dataset_num_exclude']}-" \
                                f"dsUS_{self.configs['dataset_use_scaled']}-" \
                                f"dsUTS_{self.configs['dataset_use_tree_structure']}-" \
                                f"model_{model_name}-" \
                                f"modelLS_{self.configs['siamese_latent_size']}-" \
                                f"modelOFS_{self.configs['siamese_out_fc_size']}-" \
                                f"modelLDOUT_{self.configs['siamese_latent_dropout']}-" \
                                f"bnlstmHS_{self.configs['bnlstm_hidden_size']}-" \
                                f"bnlstmNL{self.configs['bnlstm_n_layers']}-" \
                                f"bnlstmDOUT_{self.configs['bnlstm_dropout']}-" \
                                f"loss_{loss_name}-" \
                                f"{aux_loss_log}" 
        '''
        f"optLR_{self.configs['optimizer_base_lr']}-" \
        f"optWD_{self.configs['optimizer_weight_decay']}-" \
        f"sched_{self.lr_scheduler.__class__.__name__}-" \
        f"schedMS_{self.configs['scheduler_milestones']}-" \
        f"schedGAM_{self.configs['scheduler_gamma']}"
        '''
        
        logger = TensorBoardLogger(save_dir=save_dir, name=self.logger_subdir, version=self.logger_run_name, default_hp_metric=False, log_graph=False)
        
        return logger
    


    def initialize_checkpoint_callback(self):

        checkpoint_callback = ModelCheckpoint(
            monitor="val_acc_50_epoch",
            dirpath=f"model_checkpoints/" \
                    f"{self.logger_subdir}/" \
                    f"{self.logger_run_name}",
            filename='{epoch:02d}-{val_acc_50_epoch:.4f}',
            save_top_k=self.configs["train_num_epochs"],
            mode='max',
        )

        return checkpoint_callback



if __name__ == "__main__":

    model = SiameseModel()
    logger = model.initialize_logger()
    checkpoint_callback = model.initialize_checkpoint_callback()
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    trainer = Trainer(gpus=1, deterministic=True, max_epochs=model.configs['train_num_epochs'], callbacks=[checkpoint_callback, lr_monitor], logger=logger, fast_dev_run=False)
    trainer.fit(model)