import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from datasets import SupervisedDataset, SiameseDataset
from models import LSTM, LogisticRegression, SiameseNetwork



class SupervisedTest:
    def __init__(self, window_size=3) -> None:
        self.window_size = window_size

        self.dataset = SupervisedDataset(mode='test', window_size=self.window_size, log_reg=False)
        self.loader = DataLoader(self.dataset, batch_size=64, shuffle=False)
        
        self.checkpoint_path = "./model_params/val/blstm_bs64_lr1e-3_ws40_hs128_nl2_dout50/val_lstm_epoch91_acc81.9593.pth"
        #self.checkpoint_path = "./model_params/logistic_regression/val/val_logreg_epoch100_acc35.8229.pth"
        self.checkpoint = torch.load(self.checkpoint_path)
        
        self.model = LSTM(input_size=78, hidden_size=128, num_classes=170, n_layers=2).to(device=torch.device('cuda:0'))
        #self.model = LogisticRegression(num_keypoints=78, num_features=2, num_classes=170).to(device=torch.device('cuda:0'))
        self.model.load_state_dict(self.checkpoint, strict=True)
        self.model.eval()

        #self.criterion = nn.BCEWithLogitsLoss()     # Use this for Logistic Regression training
        self.criterion = nn.BCELoss()              # Use this for LSTM training (with Softmax)

        #self.writer_text = SummaryWriter('./Tensorboard/test_text/')
        #self.writer_avg_test_loss = SummaryWriter('./Tensorboard/test_loss/')
        #self.writer_hparams = SummaryWriter('./Tensorboard/test_hparams/')



    def _start_batch_test(self) -> None:
        current_iter                      = 0
        running_loss                      = 0.0
        average_loss                      = 0.0

        num_data                          = 0

        running_correct_preds             = 0
        running_correct_classwise_preds   = [0] * 170
        running_false_classwise_preds     = [0] * 170
        running_all_classwise_gt_labels      = [0] * 170

        with torch.no_grad():
            for batch_window, batch_label in self.loader:
                current_iter += 1
                
                outs = self.model(batch_window)

                loss = self.criterion(outs, batch_label)

                running_loss += loss.item()
                average_loss = running_loss / current_iter

                pred_confidence, pred_index = torch.max(outs, dim=1)
                gt_confidence, gt_index = torch.max(batch_label, dim=1)

                #batch_correct_preds = torch.eq(pred_index, gt_index).long().sum().item()
                #batch_accuracy = (batch_correct_preds / batch_window.shape[0]) * 100

                num_data += batch_window.shape[0]
                

                batch_accuracy, batch_correct_preds, classwise_correct_preds, classwise_false_preds, classwise_gt_labels = self._calculate_batch_accuracy(outs, batch_label)
                running_correct_preds += batch_correct_preds
                running_correct_classwise_preds = self._add_lists_elementwise(running_correct_classwise_preds, classwise_correct_preds)
                running_false_classwise_preds = self._add_lists_elementwise(running_false_classwise_preds, classwise_false_preds)
                running_all_classwise_gt_labels = self._add_lists_elementwise(running_all_classwise_gt_labels, classwise_gt_labels)

                if current_iter % 1 == 0:
                    print(f"\nITER#{current_iter} BATCH TEST ACCURACY: {batch_accuracy:.4f}, RUNNING TEST LOSS: {loss.item():.8f}")
                    print(f"Predicted / GT index:\n{pred_index}\n{gt_index}\n")

                
            #epoch_accuracy = (running_correct_preds / num_data) * 100
            epoch_accuracy, classwise_accuracy = self._calculate_epoch_accuracy(running_correct_preds, running_correct_classwise_preds, running_all_classwise_gt_labels, num_data)
            print(f"\n\nTEST WINDOW-WISE ACCURACY: {epoch_accuracy:.4f}, AVERAGE TEST LOSS: {average_loss:.8f}\n\n")
            
            correct_vid = 0
            false_vid = 0
            for i in range(len(running_correct_classwise_preds)):
                print(f"Person{i:03d} | Number of correct/all predictions: {running_correct_classwise_preds[i]:<3d}/{running_all_classwise_gt_labels[i]:<5d} | Accuracy: {classwise_accuracy[i]:.2f}%")
                if (running_correct_classwise_preds[i] + running_false_classwise_preds[i]) != 0:
                    if classwise_accuracy[i] >= 50:
                        correct_vid += 1
                    else:
                        false_vid += 1
                
            videowise_accuracy = (correct_vid/(correct_vid + false_vid))*100
            print(f"\n\nTEST VIDEO-WISE ACCURACY: {videowise_accuracy:.4f}%\n\n")

    def start(self, mode="batch") -> None:
        if mode == "batch":
            self._start_batch_test()



    def _calculate_batch_accuracy(self, predictions, annotations):

        pred_confidence, pred_index = torch.max(predictions, dim=1)
        gt_confidence, gt_index     = torch.max(annotations, dim=1)
        
        person1 = 1
        if gt_index[0] == 0:
            person1 += gt_index.shape[0]
        
        batch_correct_preds         = torch.eq(pred_index, gt_index).long().sum().item()
        
        batch_accuracy              = (batch_correct_preds / predictions.shape[0]) * 100

        # Calculating number of classwise correct/false predictions
        classwise_correct_preds     = torch.zeros(170).long()
        classwise_false_preds       = torch.zeros(170).long()
        classwise_gt_labels         = torch.zeros(170).long()

        correct_preds_class         = pred_index[torch.eq(pred_index, gt_index)].long()
        false_preds_class           = pred_index[torch.ne(pred_index, gt_index)].long()
        
        for element in correct_preds_class:
            classwise_correct_preds[element] += 1
        
        for element in false_preds_class:
            classwise_false_preds[element] += 1

        for element in gt_index:
            classwise_gt_labels[element] += 1

        classwise_correct_preds     = classwise_correct_preds.tolist()
        classwise_false_preds       = classwise_false_preds.tolist()

        return batch_accuracy, batch_correct_preds, classwise_correct_preds, classwise_false_preds, classwise_gt_labels



    def _add_lists_elementwise(self, list1, list2):
        array1 = np.array(list1)
        array2 = np.array(list2)

        sum_list = (array1 + array2).tolist()

        return sum_list
    

    
    def _calculate_epoch_accuracy(self, running_correct_preds, running_correct_classwise_preds, running_all_classwise_gt_labels, num_data):
        epoch_accuracy = (running_correct_preds / num_data) * 100

        classwise_accuracy = [0] * 170 #(((np.array(running_correct_classwise_preds) / (running_correct_classwise_preds + running_false_classwise_preds))) * 100).tolist()
        for i in range(len(running_correct_classwise_preds)):
            if (running_all_classwise_gt_labels[i]) == 0:
                classwise_accuracy[i] = 0
            else:
                classwise_accuracy[i] = (running_correct_classwise_preds[i] / running_all_classwise_gt_labels[i]) * 100    

        return epoch_accuracy, classwise_accuracy



class SiameseTest:
    def __init__(self) -> None:
        self.now = datetime.now().strftime("%Y.%m.%d-%H:%M:%S")
        self.device = torch.device('cuda:0')

        # Initializing window dataset hiperparams
        self.window_size = 40
        self.num_test_person = 20

        # Initializing window datasets
        self.train_window_dataset = SiameseDataset(dataset='train', is_train=False, window_size=self.window_size, num_test_person=self.num_test_person).base_dataset
        self.val_window_dataset = SiameseDataset(dataset="val", is_train=False, window_size=self.window_size, num_test_person=self.num_test_person).base_dataset
        self.test_window_dataset = SiameseDataset(dataset='test', is_train=False, window_size=self.window_size).base_dataset

        # Initializing the trained model
        self.model_checkpoint = 'model_params/siamese_network/version_5_bs_256_baselr_1e-3_multisteplr_milestones_5-15-25_gamma_0.33_lstoutsize_170/version-5-siamese-epoch=28-val_acc_epoch=0.9355.ckpt'
        self.model = SiameseNetwork().load_from_checkpoint(checkpoint_path=self.model_checkpoint).to(device=self.device).eval()

        # Creating core database
        self.averaging_types = ['mean', 'median']
        self.averaging_type = self.averaging_types[0]
        #self.database = self._create_database()

        self.metric_method = 'micro'
        self.similarity_thres = 0.95

        self.file_log_path = f'test_metrics_file_logs/{self.now}-simthres_{self.similarity_thres}-avgtype_{self.averaging_type}/'
        self.metrics_file_path = self.file_log_path + 'metrics.txt'

         

    def _create_database(self):
        database = torch.zeros(len(self.train_window_dataset), self.model.latent_size)

        with torch.no_grad():            
            for i, person_windows in enumerate(tqdm(self.train_window_dataset, desc="Creating core database (using the training dataset)", unit=" person")):
                x = torch.as_tensor(person_windows, dtype=torch.float32).to(device=self.device)
                label = self.model(x, mode='create_labels')

                if self.averaging_type == 'median':
                    database[i] = torch.median(label, dim=0)[0]
                elif self.averaging_type == 'mean':
                    database[i] = torch.mean(label, dim=0)
                else:
                    raise NotImplementedError
        
        return database
    


    def start(self):
        for similarity_thres in [0.9, 0.95, 0.98]:#[0.5, 0.8, 0.85, 0.9, 0.95, 0.98]:
            for averaging_type in self.averaging_types:
                print(f'\n\nTesting with [similarity threshold = {similarity_thres}] | [averaging type = {averaging_type}] ...')

                self.similarity_thres = similarity_thres
                self.averaging_type = averaging_type

                self.database = self._create_database()
                self.file_log_path = f'test_metrics_file_logs/{self.now}-simthres_{self.similarity_thres}-avgtype_{self.averaging_type}/'
                self.metrics_file_path = self.file_log_path + 'metrics.txt'


                self._calculate_similarity(window_dataset=self.train_window_dataset, dataset_type='training')
                self._calculate_similarity(window_dataset=self.val_window_dataset, dataset_type='validation')
                self._calculate_similarity(window_dataset=self.test_window_dataset, dataset_type='test')

    

    def _calculate_similarity(self, window_dataset, dataset_type):
        database = torch.zeros(len(window_dataset), self.model.latent_size)

        sum_accuracy = 0.0
        sum_precision = 0.0
        sum_recall = 0.0
        sum_f1_score = 0.0
        sum_cm = None

        with torch.no_grad():            
            for i, person_windows in enumerate(tqdm(window_dataset, desc=f"Creating temporary database (using the {dataset_type} dataset)", unit=" person")):
                x = torch.as_tensor(person_windows, dtype=torch.float32).to(device=self.device)
                label = self.model(x, mode='create_labels')

                if self.averaging_type == 'median':
                    database[i] = torch.median(label, dim=0)[0]
                elif self.averaging_type == 'mean':
                    database[i] = torch.mean(label, dim=0)
                else:
                    raise NotImplementedError

            num_iter = 0
            for i in tqdm(range(0, database.shape[0]), desc=f"Calculating similarities (core database - {dataset_type} dataset)", unit=" person"):
                label1 = torch.unsqueeze(database[i], dim=0).to(device=torch.device('cuda:0')) # New videos

                pos_idx = [self.database.shape[0], 0.0]
                for j in range(0, self.database.shape[0]):
                    label2 = torch.unsqueeze(self.database[j], dim=0).to(device=torch.device('cuda:0'))  # Existing videos
                    similarity_score = self.model(label1, label2, mode='test_labels').item()

                    if similarity_score > self.similarity_thres:
                        if similarity_score > pos_idx[1]:
                            pos_idx = [j, similarity_score]
                
                # Handling all new Person videos as a universal unseen class
                if i > self.database.shape[0]:
                    i = self.database.shape[0]

                sum_accuracy += accuracy_score(np.array([i]), np.array([pos_idx[0]]))
                sum_precision += precision_score(np.array([i]), np.array([pos_idx[0]]), labels=list(range(0, (self.database.shape[0] + 1))), zero_division=0, average=self.metric_method)
                sum_recall += recall_score(np.array([i]), np.array([pos_idx[0]]), labels=list(range(0, (self.database.shape[0] + 1))), zero_division=0, average=self.metric_method)
                sum_f1_score += f1_score(np.array([i]), np.array([pos_idx[0]]), labels=list(range(0, (self.database.shape[0] + 1))), zero_division=0, average=self.metric_method)
                current_cm = confusion_matrix(np.array([i]), np.array([pos_idx[0]]), labels=list(range(0, (self.database.shape[0] + 1))))
               
                if sum_cm is None:
                    sum_cm = current_cm
                else:
                    sum_cm += current_cm

                num_iter += 1

            avg_metrics = [sum_accuracy/num_iter, sum_precision/num_iter, sum_recall/num_iter, sum_f1_score/num_iter]

            self._create_cm_figure(cm=sum_cm, class_names=[f'Person{i:03d}' for i in range(0, (self.database.shape[0]))] + ['NewPerson'], dataset_type=dataset_type)
            self._log_to_file(metrics=avg_metrics, dataset_type=dataset_type)

    

    def _create_cm_figure(self, cm, class_names, dataset_type):
        """
        Returns a matplotlib figure containing the plotted confusion matrix.
        
        Args:
        cm (array, shape = [n, n]): a confusion matrix of integer classes
        class_names (array, shape = [n]): String names of the integer classes
        """
        
        figure = plt.figure(figsize=(80, 80))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        # Normalize the confusion matrix.
        #cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
        
        # Use white text if squares are dark; otherwise black.
        threshold = cm.max() / 2.
        
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
            
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        # Saving the figures
        if not os.path.exists(self.file_log_path):
            os.mkdir(self.file_log_path)
        plt.savefig(f'{self.file_log_path}/{dataset_type}_confusion_matrix.png')

        plt.close('all')


    
    def _log_to_file(self, metrics, dataset_type):
        # Logging metrics into file
        with open(self.metrics_file_path, 'a') as f:
            f.write(f'{dataset_type.capitalize()} phase\n' \
                        f'Accuracy: {metrics[0]}\n' \
                        f'Precision: {metrics[1]}\n' \
                        f'Recall: {metrics[2]}\n' \
                        f'F1 Score: {metrics[3]}\n\n')
        


if __name__=="__main__":
    #test = SupervisedTest(window_size=40)
    #test.start()

    test = SiameseTest()
    test.start()


        