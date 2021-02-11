import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import time
import warnings
from dataset import Andersson_windows_dataset, Andersson_iterating_dataset
from model import LSTM


def train(batch_size=64, window_size=3, epochs=100):

    train_windows_dataset = Andersson_windows_dataset(mode='train', window_size=window_size)
    train_windows_loader = DataLoader(train_windows_dataset, batch_size=1, shuffle=True)

    val_windows_dataset = Andersson_windows_dataset(mode='val', window_size=window_size)
    val_windows_loader = DataLoader(val_windows_dataset, batch_size=1, shuffle=True)

    base_lr_rate = 1e-2
    weight_decay = 0.000016

    model = LSTM(input_size=40, hidden_size=512, num_classes=170, n_layers=16).to(device=torch.device('cuda:0'))   

    #criterion = nn.BCEWithLogitsLoss()
    criterion = nn.CrossEntropyLoss()
    #criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=base_lr_rate, weight_decay=weight_decay, amsgrad=True)

    for current_epoch in range(epochs):

        current_train_iter                      = 0
        current_val_iter                        = 0
        
        running_train_loss                      = 0.0
        current_average_train_loss              = 0.0
        running_val_loss                        = 0.0
        current_average_val_loss                = 0.0

        num_train_data                          = 0
        num_val_data                            = 0

        running_train_correct_preds             = 0
        running_train_correct_classwise_preds   = [0] * 170

        running_val_correct_preds               = 0
        running_val_correct_classwise_preds     = [0] * 170

        for phase in ['train', 'val']:

            # Train loop
            if phase == 'train':
                train_epoch_since = time.time()

                model.train()

                for train_windows, train_track_id in train_windows_loader:

                    train_iterating_dataset = Andersson_iterating_dataset(windows=train_windows, track_id=train_track_id)
                    train_iterating_loader = DataLoader(train_iterating_dataset, batch_size=batch_size, shuffle=True)
                    #train_iterator = iter(train_iterating_loader)

                    for train_batch_window, train_batch_label in train_iterating_loader:
                        current_train_iter += 1

                        outs = model(train_batch_window)        
            
                        #scheduler = poly_lr_scheduler(optimizer = optimizer, init_lr = base_lr_rate, iter = current_iter, lr_decay_iter = 1, 
                        #                          max_iter = max_iter, power = power)                                                          # max_iter = len(train_loader)
                
                        optimizer.zero_grad()
                
                        #loss = criterion(outs, train_batch_label)
                        gt_confidence, gt_index = torch.max(train_batch_label, dim=1)
                        loss = criterion(outs, gt_index)

                        running_train_loss += loss.item()
                        current_average_train_loss = running_train_loss / current_train_iter
                    
                        loss.backward(retain_graph=False)
                
                        optimizer.step()

                        pred_confidence, pred_index = torch.max(outs, dim=1)
                        #gt_confidence, gt_index = torch.max(train_batch_label, dim=1)
                        batch_correct_preds = torch.eq(pred_index, gt_index).long().sum().item()
                        batch_accuracy = (batch_correct_preds / train_batch_window.shape[0]) * 100

                        num_train_data += train_batch_window.shape[0]
                        running_train_correct_preds += batch_correct_preds

                        if current_train_iter % 10 == 0:
                            print(f"\nITER#{current_train_iter} BATCH TRAIN ACCURACY: {batch_accuracy}, RUNNING TRAIN LOSS: {loss.item()}")
                            print(f"Predicted / GT index:\n{pred_index}\n{gt_index}\n")

                last_epoch_average_train_loss = current_average_train_loss
                epoch_accuracy = (running_train_correct_preds / num_train_data) * 100

                print(f"EPOCH#{current_epoch+1} EPOCH TRAIN ACCURACY: {epoch_accuracy}, AVERAGE TRAIN LOSS: {last_epoch_average_train_loss}")

                train_time_elapsed = time.time() - train_epoch_since
            
            # Validation loop
            elif phase == 'val':
                val_epoch_since = time.time()   
               
                model.eval()
                
                with torch.no_grad():
                    for val_windows, val_track_id in val_windows_loader:

                        val_iterating_dataset = Andersson_iterating_dataset(windows=val_windows, track_id=val_track_id)
                        val_iterating_loader = DataLoader(val_iterating_dataset, batch_size=batch_size, shuffle=True)
                        #val_iterator = iter(val_iterating_loader)
                        
                        for val_batch_window, val_batch_label in val_iterating_loader:
                            current_val_iter += 1               
                            
                            outs = model(val_batch_window)
                            
                            gt_confidence, gt_index = torch.max(val_batch_label, dim=1)
                            #val_loss = criterion(outs, val_batch_label)
                            val_loss = criterion(outs, gt_index)
                        
                            running_val_loss += val_loss.item()
                            current_average_val_loss = running_val_loss / current_val_iter

                            pred_confidence, pred_index = torch.max(outs, dim=1)
                            #gt_confidence, gt_index = torch.max(val_batch_label, dim=1)
                            batch_correct_preds = torch.eq(pred_index, gt_index).long().sum().item()
                            batch_accuracy = (batch_correct_preds / val_batch_window.shape[0]) * 100

                            num_val_data += val_batch_window.shape[0]
                            running_val_correct_preds += batch_correct_preds

                            if current_val_iter % 10 == 0:
                                print(f"ITER#{current_val_iter} BATCH VALIDATION ACCURACY: {batch_accuracy}, RUNNING VALIDATION LOSS: {val_loss.item()}")
                                print(f"Predicted / GT index: {pred_index} / {gt_index}\n")

                    last_epoch_average_val_loss = current_average_val_loss
                    epoch_accuracy = (running_val_correct_preds / num_val_data) * 100
                    print(f"EPOCH#{current_epoch+1} EPOCH VALIDATION ACCURACY: {epoch_accuracy}, AVERAGE VALIDATION LOSS: {last_epoch_average_val_loss}")

                    val_time_elapsed = time.time() - val_epoch_since


if __name__ == "__main__":
    train(batch_size=64, window_size=80, epochs=100)