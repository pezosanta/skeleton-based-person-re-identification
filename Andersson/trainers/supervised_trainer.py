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

from models.logistic_regression import LogisticRegression
from models.bnlstm import BNLSTM
from models.lstm import LSTM
from datasets import DatasetType, TrainingMode, SupervisedDataset
from utils import create_cm_figure



class SupervisedModel(LightningModule):
    def __init__(self):
        super(SupervisedModel, self).__init__()

        self.device_type = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # For reproducibility purposes
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        self.now = datetime.now().strftime('%Y.%m.%d-%H:%M:%S')

        self.configs = yaml.safe_load(open('trainers/supervised_trainer_config.yml').read())
        
        
        self.train_dataset = SupervisedDataset(
            dataset_type=DatasetType.TRAINING,
            window_size=self.configs['dataset_window_size'],
            shift_size=self.configs['dataset_shift_size'],
            num_exclude=self.configs['dataset_num_exclude'],
            mode=TrainingMode.SUPERVISED,
            use_tree_structure=self.configs['dataset_use_tree_structure'],
            use_scaled_dataset=self.configs['dataset_use_scaled']
        )
        
        self.val_dataset = SupervisedDataset(
            dataset_type=DatasetType.VALIDATION,
            window_size=self.configs['dataset_window_size'],
            shift_size=self.configs['dataset_shift_size'],
            num_exclude=self.configs['dataset_num_exclude'],
            mode=TrainingMode.SUPERVISED,
            use_tree_structure=self.configs['dataset_use_tree_structure'],
            use_scaled_dataset=self.configs['dataset_use_scaled']
        )
        
        
        self.model = BNLSTM(
            input_size=self.train_dataset.dataset.shape[2],
            lstm_output_size=self.train_dataset.windows_num.shape[0],
            hidden_size=self.configs['bnlstm_hidden_size'],
            n_layers=self.configs['bnlstm_n_layers'],
            dropout=self.configs['bnlstm_dropout']
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

        self.loss_weight = self.train_dataset.get_class_weights() if self.configs['loss_use_weight'] else None
        self.criterion = nn.CrossEntropyLoss(weight=self.loss_weight)
        #self.criterion = nn.BCEWithLogitsLoss(weight=self.loss_weight)

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

                    f"Model: {self.model.__class__.__name__}\n\n" \
                    f"Model hidden size: {self.configs['bnlstm_hidden_size']}\n\n" \
                    f"Model number of layers: {self.configs['bnlstm_n_layers']}\n\n" \
                    f"Model dropout value: {self.configs['bnlstm_dropout']}\n\n" \
                    f"Model number of trainable parameters: {self.model.count_parameters()}\n\n" \
                    
                    f"Loss type: {self.criterion.__class__.__name__}\n\n" \
                    f"Loss using class weights: {self.configs['loss_use_weight']}\n\n" \

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

        #y_one_hot = F.one_hot(y, self.train_dataset.windows_num.shape[0]).to(dtype=torch.float32)
        #loss = self.criterion(y_hat, y_one_hot)
        

        # Calculating metrics
        y_hat = torch.argmax(y_hat, dim=1)

        acc = accuracy_score(y.cpu().view(-1), y_hat.cpu().view(-1))
        prec = precision_score(y.cpu().view(-1), y_hat.cpu(), labels=list(range(self.train_dataset.windows_num.shape[0])), zero_division=0, average='macro')
        recall = recall_score(y.cpu().view(-1), y_hat.cpu(), labels=list(range(self.train_dataset.windows_num.shape[0])), zero_division=0, average='macro')
        f1 = f1_score(y.cpu().view(-1), y_hat.cpu(), labels=list(range(self.train_dataset.windows_num.shape[0])), zero_division=0, average='macro')
        cm = confusion_matrix(y.cpu().view(-1), y_hat.cpu(), labels=list(range(self.train_dataset.windows_num.shape[0])))


        # Logging to Terminal 
        self.log('train_loss', loss, on_step=False, on_epoch=False, prog_bar=False, logger=False)
        self.log('train_acc', acc, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        self.log('train_prec', prec, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        self.log('train_recall', recall, on_step=True, on_epoch=False, prog_bar=True, logger=False)
        self.log('train_f1', f1, on_step=True, on_epoch=False, prog_bar=True, logger=False)

        #return {'loss': loss, 'acc': acc, 'prec': prec, 'recall': recall, 'f1_score': f1, 'cm': cm}
        return {'loss': loss, 'y_hat': y_hat, 'y': y}
    


    def training_epoch_end(self, outputs) -> None:
        epoch_loss = torch.stack([out['loss'] for out in outputs]).mean()
        all_y_hat = torch.stack([out['y_hat'] for out in outputs]).view(-1)
        all_y = torch.stack([out['y'] for out in outputs]).view(-1)

        epoch_acc = accuracy_score(all_y.cpu(), all_y_hat.cpu())
        epoch_prec = precision_score(all_y.cpu(), all_y_hat.cpu(), labels=list(range(self.train_dataset.windows_num.shape[0])), zero_division=0, average='macro')
        epoch_recall = recall_score(all_y.cpu(), all_y_hat.cpu(), labels=list(range(self.train_dataset.windows_num.shape[0])), zero_division=0, average='macro')
        epoch_f1 = f1_score(all_y.cpu(), all_y_hat.cpu(), labels=list(range(self.train_dataset.windows_num.shape[0])), zero_division=0, average='macro')
        epoch_cm = confusion_matrix(all_y.cpu(), all_y_hat.cpu(), labels=list(range(self.train_dataset.windows_num.shape[0])))
        
        # Logging metrics to Tensorboard
        self.logger.experiment.add_scalar(tag='Training Loss', scalar_value=(epoch_loss), global_step=self.current_epoch)
        self.logger.experiment.add_scalar(tag='Training Accuracy', scalar_value=(epoch_acc), global_step=self.current_epoch)
        self.logger.experiment.add_scalar(tag='Training Precision', scalar_value=(epoch_prec), global_step=self.current_epoch)
        self.logger.experiment.add_scalar(tag='Training Recall', scalar_value=(epoch_recall), global_step=self.current_epoch)
        self.logger.experiment.add_scalar(tag='Training F1 Score', scalar_value=(epoch_f1), global_step=self.current_epoch)

        # Logging confusion matrix to Tensorboard only at every self.configs['train_cm_epoch_interval'] epochs
        if self.current_epoch == 1 or ((self.current_epoch % self.configs['train_cm_epoch_interval'] == 0) and (self.current_epoch > 0)):
            cm_figure = create_cm_figure(cm=epoch_cm, class_names=[f'Person{i:03d}' for i in range(self.train_dataset.windows_num.shape[0])])
            self.logger.experiment.add_figure(tag=f'Training Confusion Matrix/Epoch-{self.current_epoch}', figure=cm_figure, global_step=self.current_epoch)
            plt.close('all')



    def validation_step(self, batch, batch_idx):
        x, y = batch[0].to(device=self.device_type), batch[1].to(device=self.device_type)    
        
        y_hat = self.forward(x)
        loss = self.criterion(y_hat, y)

        #y_one_hot = F.one_hot(y, self.train_dataset.windows_num.shape[0]).to(dtype=torch.float32)
        #loss = self.criterion(y_hat, y_one_hot)
        

        # Calculating metrics
        y_hat = torch.argmax(y_hat, dim=1)

        acc = accuracy_score(y.cpu().view(-1), y_hat.cpu().view(-1))

        # Logging validation accuracy fot checkpoint callback
        self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=False)

        #return {'loss': loss, 'acc': acc, 'prec': prec, 'recall': recall, 'f1_score': f1, 'cm': cm}
        return {'loss': loss, 'y_hat': y_hat, 'y': y}



    def validation_epoch_end(self, outputs) -> None:
        epoch_loss = torch.stack([out['loss'] for out in outputs]).mean()
        all_y_hat = torch.stack([out['y_hat'] for out in outputs]).view(-1)
        all_y = torch.stack([out['y'] for out in outputs]).view(-1)

        epoch_acc = accuracy_score(all_y.cpu(), all_y_hat.cpu())
        epoch_prec = precision_score(all_y.cpu(), all_y_hat.cpu(), labels=list(range(self.train_dataset.windows_num.shape[0])), zero_division=0, average='macro')
        epoch_recall = recall_score(all_y.cpu(), all_y_hat.cpu(), labels=list(range(self.train_dataset.windows_num.shape[0])), zero_division=0, average='macro')
        epoch_f1 = f1_score(all_y.cpu(), all_y_hat.cpu(), labels=list(range(self.train_dataset.windows_num.shape[0])), zero_division=0, average='macro')
        epoch_cm = confusion_matrix(all_y.cpu(), all_y_hat.cpu(), labels=list(range(self.train_dataset.windows_num.shape[0])))

        
        # Logging metrics to Tensorboard
        self.logger.experiment.add_scalar(tag='Validation Loss', scalar_value=(epoch_loss), global_step=self.current_epoch)
        self.logger.experiment.add_scalar(tag='Validation Accuracy', scalar_value=(epoch_acc), global_step=self.current_epoch)
        self.logger.experiment.add_scalar(tag='Validation Precision', scalar_value=(epoch_prec), global_step=self.current_epoch)
        self.logger.experiment.add_scalar(tag='Validation Recall', scalar_value=(epoch_recall), global_step=self.current_epoch)
        self.logger.experiment.add_scalar(tag='Validation F1 Score', scalar_value=(epoch_f1), global_step=self.current_epoch)
    

        # Logging confusion matrix to Tensorboard only at every self.configs['train_cm_epoch_interval'] epochs
        if self.current_epoch == 1 or ((self.current_epoch % self.configs['train_cm_epoch_interval'] == 0) and (self.current_epoch > 0)):
            cm_figure = create_cm_figure(cm=epoch_cm, class_names=[f'Person{i:03d}' for i in range(self.train_dataset.windows_num.shape[0])])
            self.logger.experiment.add_figure(tag=f'Validation Confusion Matrix/Epoch-{self.current_epoch}', figure=cm_figure, global_step=self.current_epoch)
            plt.close('all')


    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.configs['train_batch_size'], shuffle=True, drop_last=True, num_workers=0)



    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.configs['train_batch_size'], shuffle=False, drop_last=True, num_workers=0) 

    
    
    def initialize_logger(self):
        model_name = f"{self.model.__class__.__name__}" if self.model.__class__.__name__ != "LogisticRegression" else "LOGREG"
        loss_name = f"{self.criterion.__class__.__name__}" if self.criterion.__class__.__name__ != "BCEWithLogitsLoss" else "BCEWLL"

        save_dir = 'tensorboard_logs'
        self.logger_subdir = 'supervised'
        self.logger_run_name =  f"{self.now}-" \
                                f"epochs_{self.configs['train_num_epochs']}-" \
                                f"bs_{self.configs['train_batch_size']}-" \
                                f"dsWS_{self.configs['dataset_window_size']}-" \
                                f"dsSS_{self.configs['dataset_shift_size']}-" \
                                f"dsNE_{self.configs['dataset_num_exclude']}-" \
                                f"dsUS_{self.configs['dataset_use_scaled']}-" \
                                f"dsUTS_{self.configs['dataset_use_tree_structure']}-" \
                                f"model_{model_name}-" \
                                f"modelHS_{self.configs['bnlstm_hidden_size']}-" \
                                f"modelNL{self.configs['bnlstm_n_layers']}-" \
                                f"modelDOUT_{self.configs['bnlstm_dropout']}-" \
                                f"loss_{loss_name}-" \
                                f"lossUW_{self.configs['loss_use_weight']}-" \
                                f"opt_{self.optimizer.__class__.__name__}-" \
                                f"optLR_{self.configs['optimizer_base_lr']}-" \
                                f"optWD_{self.configs['optimizer_weight_decay']}-" \
                                f"sched_{self.lr_scheduler.__class__.__name__}-" \
                                f"schedMS_{self.configs['scheduler_milestones']}-" \
                                f"schedGAM_{self.configs['scheduler_gamma']}"
        
        logger = TensorBoardLogger(save_dir=save_dir, name=self.logger_subdir, version=self.logger_run_name, default_hp_metric=False, log_graph=False)
        
        return logger
    


    def initialize_checkpoint_callback(self):

        checkpoint_callback = ModelCheckpoint(
            monitor="val_acc_epoch",
            dirpath=f"model_checkpoints/" \
                    f"{self.logger_subdir}/" \
                    f"{self.logger_run_name}",
            filename='{epoch:02d}-{val_acc_epoch:.4f}',
            save_top_k=self.configs["train_num_epochs"],
            mode='max',
        )

        return checkpoint_callback



if __name__ == '__main__':
    
    model = SupervisedModel()
    logger = model.initialize_logger()
    checkpoint_callback = model.initialize_checkpoint_callback()
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = Trainer(gpus=1, deterministic=True, max_epochs=model.configs['train_num_epochs'], callbacks=[checkpoint_callback, lr_monitor], logger=logger, fast_dev_run=False)
    trainer.fit(model)
    