import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import time
import warnings
from dataset import Andersson_dataset
from models import LSTM, LogisticRegression



class Test:
    def __init__(self, window_size=3) -> None:
        self.window_size = window_size

        self.dataset = Andersson_dataset(mode='test', window_size=self.window_size, log_reg=False)
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



if __name__=="__main__":
    test = Test(window_size=40)
    test.start()


        