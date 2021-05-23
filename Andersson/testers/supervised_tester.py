import os
import random
import yaml
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from models.logistic_regression import LogisticRegression
from models.lstm import LSTM
from models.bnlstm import BNLSTM
from models.self_supervised import SelfSupervised
from datasets import DatasetType, TrainingMode, SupervisedDataset
from utils import create_cm_figure, create_metrics_figure



class SupervisedTester():
    def __init__(self) -> None:
        self.device_type = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # For reproducibility purposes
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        self.configs = yaml.safe_load(open('testers/supervised_tester_config.yml').read())

        self.dataset = SupervisedDataset(
            dataset_type=DatasetType.TEST,
            window_size=self.configs['dataset_window_size'],
            shift_size=self.configs['dataset_shift_size'],
            num_exclude=self.configs['dataset_num_exclude'],
            mode=TrainingMode.SUPERVISED,
            use_tree_structure=self.configs['dataset_use_tree_structure'],
            use_scaled_dataset=self.configs['dataset_use_scaled']
        )
        self.loader = DataLoader(self.dataset, batch_size=self.configs['batch_size'], shuffle=False, drop_last=True)
        
        '''
        self.model = BNLSTM(
            input_size=self.dataset.dataset.shape[2],
            lstm_output_size=self.dataset.windows_num.shape[0],
            hidden_size=self.configs['bnlstm_hidden_size'],
            n_layers=self.configs['bnlstm_n_layers'],
            dropout=self.configs['bnlstm_dropout']
        )
        '''
        self.model = SelfSupervised(
            input_size=self.dataset.dataset.shape[2],
            window_size=self.configs['dataset_window_size'],
            lstm_output_size=self.dataset.windows_num.shape[0],
            hidden_size=self.configs['bnlstm_hidden_size'],
            n_layers=self.configs['bnlstm_n_layers'],
            dropout=self.configs['bnlstm_dropout'],
            supervised_mode=True
        )

        '''
        self.model = LSTM(
            input_size=self.dataset.dataset.shape[2],
            lstm_output_size=self.dataset.windows_num.shape[0],
            hidden_size=self.configs['bnlstm_hidden_size'],
            n_layers=self.configs['bnlstm_n_layers'],
            dropout=self.configs['bnlstm_dropout']
        )

        self.model = LogisticRegression(
            num_keypoints=self.dataset.dataset.shape[1],
            num_features=self.dataset.dataset.shape[2],
            output_size=self.dataset.windows_num.shape[0]
        )
        '''
        
        self.model_checkpoint_path = self.configs['checkpoint_paths'][-1] 
        self.model_checkpoint = self._prepare_checkpoint()

        self.model.load_state_dict(self.model_checkpoint, strict=True)
        self.model.to(device=self.device_type).eval()

        self.save_eps = self.configs['save_eps']


    
    def _prepare_checkpoint(self):
        model_checkpoint = torch.load(self.model_checkpoint_path)['state_dict']

        if self.model.__class__.__name__ != 'LogisticRegression':
            # Delete loss weights from the dictionary
            model_checkpoint.pop('criterion.weight')

        # Creating a new checkpoint_state_dict with matching key names with the model_state_dict
        delete_prefix_len = len('model.')
        adapted_model_checkpoint = {key[delete_prefix_len:]: value for key, value in model_checkpoint.items()}

        return adapted_model_checkpoint

    

    def _start_testing_loop(self) -> None:
        outputs = []
        with torch.no_grad():
            for batch in tqdm(self.loader, desc="Testing"):
                x, y = batch[0].to(device=self.device_type), batch[1].to(device=self.device_type)
                
                y_hat = torch.argmax(self.model(x), dim=1)

                outputs.append({'y_hat': y_hat, 'y': y})
        
        return outputs



    def start(self, mode="batch") -> None:
        if mode == "batch":
            outputs = self._start_testing_loop()
            metrics = self._calculate_epoch_metrics(outputs=outputs)
            self._log_metrics(metrics=metrics)


    
    def _calculate_epoch_metrics(self, outputs):
        all_y_hat = torch.stack([out['y_hat'] for out in outputs]).view(-1)
        all_y = torch.stack([out['y'] for out in outputs]).view(-1) 

        labels = list(range(self.dataset.windows_num.shape[0]))

        cm = confusion_matrix(all_y.cpu(), all_y_hat.cpu(), labels=labels)

        classwise_acc = cm.diagonal() / cm.sum(axis=1)*100        
        classwise_prec = precision_score(all_y.cpu(), all_y_hat.cpu(), labels=labels, zero_division=0, average=None)*100 
        classwise_recall = recall_score(all_y.cpu(), all_y_hat.cpu(), labels=labels, zero_division=0, average=None)*100 
        classwise_f1 = f1_score(all_y.cpu(), all_y_hat.cpu(), labels=labels, zero_division=0, average=None)*100 
        
        videowise_acc = ((classwise_acc > 50.0).astype(int).sum() / len(classwise_acc))*100 
        
        micro_acc = accuracy_score(all_y.cpu(), all_y_hat.cpu())*100 
        micro_prec = precision_score(all_y.cpu(), all_y_hat.cpu(), labels=labels, zero_division=0, average='micro')*100 
        micro_recall = recall_score(all_y.cpu(), all_y_hat.cpu(), labels=labels, zero_division=0, average='micro')*100 
        micro_f1 = f1_score(all_y.cpu(), all_y_hat.cpu(), labels=labels, zero_division=0, average='micro')*100 

        macro_prec = precision_score(all_y.cpu(), all_y_hat.cpu(), labels=labels, zero_division=0, average='macro')*100 
        macro_recall = recall_score(all_y.cpu(), all_y_hat.cpu(), labels=labels, zero_division=0, average='macro')*100 
        macro_f1 = f1_score(all_y.cpu(), all_y_hat.cpu(), labels=labels, zero_division=0, average='macro')*100 

        weighted_prec = precision_score(all_y.cpu(), all_y_hat.cpu(), labels=labels, zero_division=0, average='weighted')*100 
        weighted_recall = recall_score(all_y.cpu(), all_y_hat.cpu(), labels=labels, zero_division=0, average='weighted')*100 
        weighted_f1 = f1_score(all_y.cpu(), all_y_hat.cpu(), labels=labels, zero_division=0, average='weighted')*100 
       
        return {
            'confusion_matrix': cm,
            'classwise_acc': classwise_acc,
            'classwise_prec': classwise_prec,
            'classwise_recall': classwise_recall,
            'classwise_f1': classwise_f1,
            'videowise_acc': videowise_acc,
            'micro_acc': micro_acc,
            'micro_prec': micro_prec,
            'micro_recall': micro_recall,
            'micro_f1': micro_f1,
            'macro_prec': macro_prec,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'weighted_prec': weighted_prec,
            'weighted_recall': weighted_recall,
            'weighted_f1': weighted_f1
        }

    

    def _log_metrics(self, metrics):
        dir_path = f"test_logs/supervised/{self.model_checkpoint_path.rsplit('/', 2)[1]}/{self.model_checkpoint_path.rsplit('/', 2)[2]}/"
        png_path = dir_path + "png/"
        eps_path = dir_path + "eps/"
        
        if not os.path.exists(path=dir_path):
            os.makedirs(png_path)
            if self.save_eps:
                os.makedirs(eps_path)
      
        '''
        cm_figure = create_cm_figure(cm=metrics['confusion_matrix'], class_names=[f'Person{i:03d}' for i in range(self.dataset.windows_num.shape[0])])
        plt.savefig(f'{png_path}confusion_matrix.png')
        if self.save_eps:
            plt.savefig(f'{eps_path}confusion_matrix.eps')
        plt.close()
        '''
        classwise_acc_figure = create_metrics_figure(metric_array=metrics['classwise_acc'], xlabel="Person indices", ylabel="Accuracy score", title=f"Class-wise accuracy score", threshold=50)
        plt.savefig(f'{png_path}classwise_acc.png')
        if self.save_eps:
            plt.savefig(f'{eps_path}classwise_acc.eps')
        plt.close()

        classwise_prec_figure = create_metrics_figure(metric_array=metrics['classwise_prec'], xlabel="Person indices", ylabel="Precision score", title=f"Class-wise precision score")
        plt.savefig(f'{png_path}classwise_prec.png')
        if self.save_eps:
            plt.savefig(f'{eps_path}classwise_prec.eps')
        plt.close()

        classwise_recall_figure = create_metrics_figure(metric_array=metrics['classwise_recall'], xlabel="Person indices", ylabel="Recall score", title=f"Class-wise recall score")
        plt.savefig(f'{png_path}classwise_recall.png')
        if self.save_eps:
            plt.savefig(f'{eps_path}classwise_recall.eps')
        plt.close()

        classwise_f1_figure = create_metrics_figure(metric_array=metrics['classwise_f1'], xlabel="Person indices", ylabel="F1 score", title=f"Class-wise F1 score")
        plt.savefig(f'{png_path}classwise_f1.png')
        if self.save_eps:
            plt.savefig(f'{eps_path}classwise_f1.eps')
        plt.close()

        # Logging metrics into file
        with open(f"{dir_path}metrics.txt", 'w') as f:
            f.write(f"{self.model_checkpoint_path.rsplit('/', 2)[1]} \n\n" \
                    f"Video-wise accuracy score: {metrics['videowise_acc']}\n" \
                    f"Micro accuracy score: {metrics['micro_acc']}\n" \
                    f"Micro precision score: {metrics['micro_prec']}\n" \
                    f"Micro recall score: {metrics['micro_recall']}\n" \
                    f"Micro F1 score: {metrics['micro_f1']}\n" \
                    f"Macro precision score: {metrics['macro_prec']}\n" \
                    f"Macro recall score: {metrics['macro_recall']}\n" \
                    f"Macro F1 score: {metrics['macro_f1']}\n" \
                    f"Weighted precision score: {metrics['weighted_prec']}\n" \
                    f"Weighted recall score: {metrics['weighted_recall']}\n" \
                    f"Weighted F1 score: {metrics['weighted_f1']}"
                    )



if __name__ == '__main__':
    tester = SupervisedTester().start()