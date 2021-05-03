import os
import random
import yaml
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from enum import Enum

import torch

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from models.bnlstm import BNLSTM
from models.siamese import ModelMode, SiameseCrossentropy
from datasets import DatasetType, SiameseDataset
from utils import create_cm_figure, create_metrics_figure






class AveragingType(Enum):
    MEAN = "mean"
    MEDIAN = "median"



class SiameseTester:
    def __init__(self) -> None:
        self.device_type = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # For reproducibility purposes
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        self.configs = yaml.safe_load(open('testers/siamese_tester_config.yml').read())
        
        # Initializing window datasets
        self.core_dataset = SiameseDataset(
            dataset_type=DatasetType.TRAINING,
            window_size=self.configs['dataset_window_size'],
            shift_size=self.configs['dataset_shift_size'],
            num_exclude=self.configs['dataset_num_exclude'],
            need_pair_dataset=False,
            use_tree_structure=self.configs['dataset_use_tree_structure'],
            use_scaled_dataset=self.configs['dataset_use_scaled']
        )

        self.train_dataset = SiameseDataset(
            dataset_type=DatasetType.TRAINING,
            window_size=self.configs['dataset_window_size'],
            shift_size=self.configs['dataset_shift_size'],
            num_exclude=0,
            need_pair_dataset=False,
            use_tree_structure=self.configs['dataset_use_tree_structure'],
            use_scaled_dataset=self.configs['dataset_use_scaled']
        )

        self.val_dataset = SiameseDataset(
            dataset_type=DatasetType.VALIDATION,
            window_size=self.configs['dataset_window_size'],
            shift_size=self.configs['dataset_shift_size'],
            num_exclude=0,
            need_pair_dataset=False,
            use_tree_structure=self.configs['dataset_use_tree_structure'],
            use_scaled_dataset=self.configs['dataset_use_scaled']
        )

        self.test_dataset = SiameseDataset(
            dataset_type=DatasetType.TEST,
            window_size=self.configs['dataset_window_size'],
            shift_size=self.configs['dataset_shift_size'],
            num_exclude=0,
            need_pair_dataset=False,
            use_tree_structure=self.configs['dataset_use_tree_structure'],
            use_scaled_dataset=self.configs['dataset_use_scaled']
        )

        # Initializing the trained model
        #self.model_checkpoint = 'model_params/siamese_network/version_5_bs_256_baselr_1e-3_multisteplr_milestones_5-15-25_gamma_0.33_lstmoutsize_170-latentsize_170-outfcsize_32/version-5-siamese-epoch=28-val_acc_epoch=0.9355.ckpt'
        #self.model_checkpoint = 'model_params/siamese_network/version_6_bs_512_baselr_3e-3_multisteplr_milestones_5-10-15-25_gamma_0.33_lstmoutsize_160-latentsize_32-outfcsize_16-abs+cos+sed/version-6-siamese-epoch=33-val_acc_epoch=0.9358.ckpt'
        #self.model = SiameseNetwork().load_from_checkpoint(checkpoint_path=self.model_checkpoint).to(device_type=self.device_type).eval()
        #self.model_checkpoint_path = "./model_checkpoints/supervised/2021.04.28-23:35:56-epochs_51-bs_128-dsWS_40-dsSS_1-dsNE_0-dsUS_True-dsUTS_True-model_BNLSTM-modelHS_256-modelNL2-modelDOUT_0.5-loss_CrossEntropyLoss-lossUW_True-opt_Adam-optLR_0.001-optWD_1e-06-sched_MultiStepLR-schedMS_[20, 40, 60, 80]-schedGAM_0.33/epoch=43-val_acc_epoch=0.8081.ckpt"
        self.model_checkpoint_path = self.configs['checkpoint_paths'][3]    
        self.model_checkpoint = self._prepare_checkpoint()
        
        core_model = BNLSTM(
                input_size=self.core_dataset.base_dataset[0].shape[2],
                lstm_output_size=self.core_dataset.windows_num.shape[0],
                hidden_size=self.configs['bnlstm_hidden_size'],
                n_layers=self.configs['bnlstm_n_layers'],
                dropout=self.configs['bnlstm_dropout']
        )

        self.model = SiameseCrossentropy(
            model=core_model,
            model_output_size=self.core_dataset.windows_num.shape[0],
            latent_size=self.configs['siamese_latent_size'],
            out_fc_size=self.configs['siamese_out_fc_size'],
            latent_fc_dropout=self.configs['siamese_latent_dropout']
        )

        self.model.load_state_dict(self.model_checkpoint, strict=True)
        self.model.to(device=self.device_type).eval()



    def _prepare_checkpoint(self):
        model_checkpoint = torch.load(self.model_checkpoint_path)['state_dict']

        # Creating a new checkpoint_state_dict with matching key names with the model_state_dict
        delete_prefix_len = len('model.')
        adapted_model_checkpoint = {key[delete_prefix_len:]: value for key, value in model_checkpoint.items()}

        return adapted_model_checkpoint

         

    def _create_database(self):
        database = torch.zeros(len(self.core_dataset.base_dataset), self.configs['siamese_latent_size']-self.model.additional_features)

        with torch.no_grad():            
            for i, person_windows in enumerate(tqdm(self.core_dataset.base_dataset, desc="Creating core database (using the training dataset)", unit=" person")):
                x = torch.as_tensor(person_windows, dtype=torch.float32).to(device=self.device_type)
                #label = self.model(x, mode='create_labels')
                label = self.model(x, mode=ModelMode.TEST_CREATING_LABELS)

                if self.averaging_type == AveragingType.MEDIAN:
                    database[i] = torch.median(label, dim=0)[0]
                elif self.averaging_type == AveragingType.MEAN:
                    database[i] = torch.mean(label, dim=0)
                else:
                    raise NotImplementedError
        
        return database
    


    def start(self):
        for similarity_thres in self.configs['similarity_thres']:
            for averaging_type in list(AveragingType):
                print(f'\n\nTesting with [similarity threshold = {similarity_thres}] | [averaging type = {averaging_type.value}] ...')

                self.similarity_thres = similarity_thres
                self.averaging_type = averaging_type

                self.database = self._create_database()

                train_results = self._calculate_similarity(window_dataset=self.train_dataset.base_dataset, dataset_type=DatasetType.TRAINING)
                val_results = self._calculate_similarity(window_dataset=self.val_dataset.base_dataset, dataset_type=DatasetType.VALIDATION)
                test_results = self._calculate_similarity(window_dataset=self.test_dataset.base_dataset, dataset_type=DatasetType.TEST)

                train_metrics = self._calculate_metrics(outputs=train_results)
                val_metrics = self._calculate_metrics(outputs=val_results)
                test_metrics = self._calculate_metrics(outputs=test_results)

                self._log_metrics(metrics=train_metrics, dataset_type=DatasetType.TRAINING)
                self._log_metrics(metrics=val_metrics, dataset_type=DatasetType.VALIDATION)
                self._log_metrics(metrics=test_metrics, dataset_type=DatasetType.TEST)

    

    def _calculate_similarity(self, window_dataset, dataset_type):
        database = torch.zeros(len(window_dataset), self.configs['siamese_latent_size']-self.model.additional_features)

        outputs = []

        with torch.no_grad():            
            for i, person_windows in enumerate(tqdm(window_dataset, desc=f"Creating temporary database (using the {dataset_type.value} dataset)", unit=" person")):
                x = torch.as_tensor(person_windows, dtype=torch.float32).to(device=self.device_type)
                label = self.model(x, mode=ModelMode.TEST_CREATING_LABELS)

                if self.averaging_type == AveragingType.MEDIAN:
                    database[i] = torch.median(label, dim=0)[0]
                elif self.averaging_type == AveragingType.MEAN:
                    database[i] = torch.mean(label, dim=0)
                else:
                    raise NotImplementedError

            for i in tqdm(range(database.shape[0]), desc=f"Calculating similarities (core database - {dataset_type.value} dataset)", unit=" person"):
                label1 = torch.unsqueeze(database[i], dim=0).to(device=self.device_type) # New videos

                pos_idx = [self.database.shape[0], 0.0]
                for j in range(self.database.shape[0]):
                    label2 = torch.unsqueeze(self.database[j], dim=0).to(device=self.device_type)  # Existing videos
                    similarity_score = self.model(label1, label2, mode=ModelMode.TEST_EVALUATING_LABELS)[0].item()

                    if similarity_score > self.similarity_thres:
                        if similarity_score > pos_idx[1]:
                            pos_idx = [j, similarity_score]
                
                # Handling all new Person videos as a universal unseen class
                if i > self.database.shape[0]:
                    i = self.database.shape[0]

                outputs.append({'y_hat': torch.tensor([pos_idx[0]]), 'y': torch.tensor([i])})

        return outputs
    


    def _calculate_metrics(self, outputs):
        all_y_hat = torch.stack([out['y_hat'] for out in outputs]).view(-1)
        all_y = torch.stack([out['y'] for out in outputs]).view(-1) 
        
        labels = list(range(self.database.shape[0] + 1))

        cm = confusion_matrix(all_y.cpu(), all_y_hat.cpu(), labels=labels)

        classwise_acc = cm.diagonal() / cm.sum(axis=1)*100        
        classwise_prec = precision_score(all_y.cpu(), all_y_hat.cpu(), labels=labels, zero_division=0, average=None)*100 
        classwise_recall = recall_score(all_y.cpu(), all_y_hat.cpu(), labels=labels, zero_division=0, average=None)*100 
        classwise_f1 = f1_score(all_y.cpu(), all_y_hat.cpu(), labels=labels, zero_division=0, average=None)*100 
        
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
   

    
    def _log_metrics(self, metrics, dataset_type):
        dir_path = f"test_logs/siamese/{self.model_checkpoint_path.rsplit('/', 2)[1]}/{dataset_type.value}/simthresh_{self.similarity_thres}-avgtype_{self.averaging_type.value}/"
        png_path = dir_path + "png/"
        eps_path = dir_path + "eps/"
        
        if not os.path.exists(path=dir_path):
            os.makedirs(png_path)
            os.makedirs(eps_path)
        
        cm_figure = create_cm_figure(cm=metrics['confusion_matrix'], class_names=[f'Person{i:03d}' for i in range(self.database.shape[0])] + ['NewPerson'])
        plt.savefig(f'{png_path}confusion_matrix.png')
        plt.savefig(f'{eps_path}confusion_matrix.eps')
        plt.close()
        
        classwise_acc_figure = create_metrics_figure(metric_array=metrics['classwise_acc'], xlabel="Person indices", ylabel="Accuracy score", title=f"Classwise accuracy score", newperson=True)
        plt.savefig(f'{png_path}classwise_acc.png')
        plt.savefig(f'{eps_path}classwise_acc.eps')
        plt.close()

        classwise_prec_figure = create_metrics_figure(metric_array=metrics['classwise_prec'], xlabel="Person indices", ylabel="Precision score", title=f"Classwise precision score", newperson=True)
        plt.savefig(f'{png_path}classwise_prec.png')
        plt.savefig(f'{eps_path}classwise_prec.eps')
        plt.close()

        classwise_recall_figure = create_metrics_figure(metric_array=metrics['classwise_recall'], xlabel="Person indices", ylabel="Recall score", title=f"Classwise recall score", newperson=True)
        plt.savefig(f'{png_path}classwise_recall.png')
        plt.savefig(f'{eps_path}classwise_recall.eps')
        plt.close()

        classwise_f1_figure = create_metrics_figure(metric_array=metrics['classwise_f1'], xlabel="Person indices", ylabel="F1 score", title=f"Classwise F1 score", newperson=True)
        plt.savefig(f'{png_path}classwise_f1.png')
        plt.savefig(f'{eps_path}classwise_f1.eps')
        plt.close()

        # Logging metrics into file
        with open(f"{dir_path}metrics.txt", 'w') as f:
            f.write(f"{self.model_checkpoint_path.rsplit('/', 2)[1]} \n\n" \
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
    tester = SiameseTester().start()
    