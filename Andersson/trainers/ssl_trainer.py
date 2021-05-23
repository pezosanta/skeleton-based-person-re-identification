import yaml
import random
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Subset, DataLoader

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

from models.self_supervised import SelfSupervised
from datasets import DatasetType, TrainingMode, SelfSupervisedDataset



class SSLPretrainerModel(LightningModule):
    def __init__(self):
        super(SSLPretrainerModel, self).__init__()

        self.device_type = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # For reproducibility purposes
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        self.now = datetime.now().strftime('%Y.%m.%d-%H:%M:%S')

        self.configs = yaml.safe_load(open('trainers/ssl_trainer_config.yml').read())
        
        self.dataset = SelfSupervisedDataset(
            dataset_type=DatasetType.SSL_SELF_SUPERVISED_TRAINING,
            window_size=self.configs['dataset_window_size'],
            shift_size=self.configs['dataset_shift_size'],
            num_exclude=self.configs['dataset_num_exclude'],
            mode=TrainingMode.SUPERVISED,
            use_tree_structure=self.configs['dataset_use_tree_structure'],
            use_scaled_dataset=self.configs['dataset_use_scaled']
        )

        self.train_dataset = Subset(dataset=self.dataset, indices=list(range(int(self.configs['dataset_train_ratio']*self.dataset.dataset.shape[0]))))
        self.val_dataset = Subset(dataset=self.dataset, indices=list(range(int(self.configs['dataset_train_ratio']*self.dataset.dataset.shape[0]), self.dataset.dataset.shape[0])))        
        
        self.model = SelfSupervised(
            input_size=self.dataset.dataset.shape[2],
            window_size=self.configs['dataset_window_size'],
            lstm_output_size=self.dataset.windows_num.shape[0],
            hidden_size=self.configs['bnlstm_hidden_size'],
            n_layers=self.configs['bnlstm_n_layers'],
            dropout=self.configs['bnlstm_dropout'],
            supervised_mode=False
        )
        
        '''
        self.model = LSTM(
            input_size=self.train_dataset.dataset.shape[2],
            lstm_output_size=self.train_dataset.windows_num.shape[0],
            hidden_size=self.configs['bnlstm_hidden_size'],
            n_layers=self.configs['bnlstm_n_layers'],
            dropout=self.configs['bnlstm_dropout']
        )
        '''
        '''
        self.model = LogisticRegression(
            num_keypoints=self.train_dataset.dataset.shape[1],
            num_features=self.train_dataset.dataset.shape[2],
            output_size=self.train_dataset.windows_num.shape[0]
        )
        '''

        self.criterion = nn.MSELoss()

        self.optimizer = optim.Adam(self.parameters(), lr=self.configs['optimizer_base_lr'], weight_decay=self.configs['optimizer_weight_decay'])
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=self.optimizer, milestones=self.configs['scheduler_milestones'], gamma=self.configs['scheduler_gamma'])

       

    def forward(self, x):
        x = self.model(x)

        return x

    

    def configure_optimizers(self):
        return [self.optimizer], [self.lr_scheduler]
    


    def on_train_start(self):
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
                    f"Dataset training size: {len(self.train_dataset)}\n\n" \
                    f"Dataset validation size: {len(self.val_dataset)}\n\n" \

                    f"Model: {self.model.__class__.__name__}\n\n" \
                    f"Model hidden size: {self.configs['bnlstm_hidden_size']}\n\n" \
                    f"Model number of layers: {self.configs['bnlstm_n_layers']}\n\n" \
                    f"Model dropout value: {self.configs['bnlstm_dropout']}\n\n" \
                    f"Model number of trainable parameters: {self.model.count_parameters()}\n\n" \
                    
                    f"Loss type: {self.criterion.__class__.__name__}\n\n" \

                    f"Optimizer type: {self.optimizer.__class__.__name__}\n\n" \
                    f"Optimizer base learning rate: {self.configs['optimizer_base_lr']}\n\n" \
                    f"Optimizer weight decay: {self.configs['optimizer_weight_decay']}\n\n" \

                    f"Scheduler type: {self.lr_scheduler.__class__.__name__}\n\n" \
                    f"Scheduler milestones: {self.configs['scheduler_milestones']}\n\n" \
                    f"Scheduler gamma: {self.configs['scheduler_gamma']}\n\n",

                global_step=0)    

    
   
    def training_step(self, batch, batch_idx):
        x, y = batch[0].to(device=self.device_type), batch[1].to(device=self.device_type)

        if self.current_epoch == 0 and batch_idx == 0:
            self.logger.experiment.add_graph(model=self.model, input_to_model=x, verbose=False)   
        
        
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)

        # Logging to Terminal 
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=False)

        return {'loss': loss}
    


    def training_epoch_end(self, outputs) -> None:
        epoch_loss = torch.stack([out['loss'] for out in outputs]).mean()
        
        # Logging metrics to Tensorboard
        self.logger.experiment.add_scalar(tag='Training Loss', scalar_value=(epoch_loss), global_step=self.current_epoch)



    def validation_step(self, batch, batch_idx):
        x, y = batch[0].to(device=self.device_type), batch[1].to(device=self.device_type)    
        
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)


        # Logging validation accuracy fot checkpoint callback
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=False)

        return {'loss': loss}



    def validation_epoch_end(self, outputs) -> None:
        epoch_loss = torch.stack([out['loss'] for out in outputs]).mean()
        
        # Logging metrics to Tensorboard
        self.logger.experiment.add_scalar(tag='Validation Loss', scalar_value=(epoch_loss), global_step=self.current_epoch)    



    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.configs['train_batch_size'], shuffle=True, drop_last=True, num_workers=0)



    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.configs['train_batch_size'], shuffle=False, drop_last=True, num_workers=0) 

    
    
    def initialize_logger(self):
        save_dir = 'tensorboard_logs'
        self.logger_subdir = 'ssl_pretraining'
        self.logger_run_name =  f"{self.now}-" \
                                f"epochs_{self.configs['train_num_epochs']}-" \
                                f"bs_{self.configs['train_batch_size']}-" \
                                f"dsWS_{self.configs['dataset_window_size']}-" \
                                f"dsSS_{self.configs['dataset_shift_size']}-" \
                                f"dsNE_{self.configs['dataset_num_exclude']}-" \
                                f"dsUS_{self.configs['dataset_use_scaled']}-" \
                                f"dsUTS_{self.configs['dataset_use_tree_structure']}-" \
                                f"model_{self.model.__class__.__name__}-" \
                                f"modelHS_{self.configs['bnlstm_hidden_size']}-" \
                                f"modelNL{self.configs['bnlstm_n_layers']}-" \
                                f"modelDOUT_{self.configs['bnlstm_dropout']}-" \
                                f"loss_{self.criterion.__class__.__name__}-" \
                                f"opt_{self.optimizer.__class__.__name__}-" \
                                f"sched_{self.lr_scheduler.__class__.__name__}"
                                
        
        logger = TensorBoardLogger(save_dir=save_dir, name=self.logger_subdir, version=self.logger_run_name, default_hp_metric=False, log_graph=False)
        
        return logger
    


    def initialize_checkpoint_callback(self):

        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss_epoch",
            dirpath=f"model_checkpoints/" \
                    f"{self.logger_subdir}/" \
                    f"{self.logger_run_name}",
            filename='{epoch:02d}-{val_loss_epoch:.4f}',
            save_top_k=self.configs["train_num_epochs"],
            mode='min',
        )

        return checkpoint_callback



if __name__ == '__main__':
    
    model = SSLPretrainerModel()
    logger = model.initialize_logger()
    checkpoint_callback = model.initialize_checkpoint_callback()
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = Trainer(gpus=1, deterministic=True, max_epochs=model.configs['train_num_epochs'], callbacks=[checkpoint_callback, lr_monitor], logger=logger, fast_dev_run=False)
    trainer.fit(model)
    