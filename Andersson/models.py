import bnlstm
from datasets import SiameseDataset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from sklearn.metrics import accuracy_score



class LSTM(nn.Module):

  # sequence length: number of subalternating input pose sequences
  # input size: number of joints (20) in a skeleton x number of coordinates (2) per joint = 40
  # hidden size: the hidden size of the LSTM
  # num_classes: the output shape of the LSTM 
  # n_layers: number of stacked LSTM modules in the LSTM  
  def __init__(self, input_size, num_classes, hidden_size=256, n_layers=16):
    super(LSTM, self).__init__()

    self.hidden_size = hidden_size
    self.n_layers = n_layers

    self.device = torch.device('cuda:0')

    self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=n_layers, batch_first=True, dropout=0.3)

    self.bnlstm = bnlstm.LSTM(cell_class=bnlstm.BNLSTMCell, input_size=input_size, hidden_size=hidden_size, num_layers=n_layers, max_length=152, batch_first=True, dropout=0.5)

    self.linear = nn.Linear(in_features=hidden_size, out_features=num_classes)

    self.softmax = nn.Softmax(dim=1)



  def _reset_hidden_state(self, batch_size):
    hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device=self.device),
              torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device=self.device))
    
    return hidden



  # Input (sequence) shape: (batch_size, sequence_length, input_size) e.g.: (64, 3, 40)
  def forward(self, sequences):
    # Reseting hidden states in every iteration
    # Hidden shape: (self.n_layers, batch_size, self.hidden_size)
    #hidden_states = self._reset_hidden_state(batch_size=sequences.shape[0])

    # LSTM output shape: (batch_size, sequence_length, self.hidden_size)
    #lstm_out, hidden_states = self.lstm(sequences, hidden_states)

    # BNLSTM output shape: (sequence_length, batch_size, self.hidden_size)
    lstm_out, (hidden_states, cell_states) = self.bnlstm(sequences)#, hx=hidden_states)
    lstm_out = lstm_out.permute(1, 0, 2)

    # Using only the output of the last sequence for prediction
    # Shape: (batch_size, self.hidden_size)
    out = lstm_out[:, -1, :]

    # Generating the output
    # Output shape: (batch_size, num_classes)
    out = self.linear(out)

    out = self.softmax(out)

    return out



class LogisticRegression(nn.Module):
  def __init__(self, num_keypoints, num_features, num_classes):
    super(LogisticRegression, self).__init__()

    self.linear = torch.nn.Linear(in_features=num_keypoints*num_features, out_features=num_classes)



  def forward(self, x):
    x = x.view(x.size(0), -1)
    x = self.linear(x)
    return x



class SiameseNetwork(LightningModule):
  def __init__(self):
    super(SiameseNetwork, self).__init__()

    # Initializing model hiperparams
    self.input_size = 78
    self.hidden_size = 128
    self.lstm_output_size = 170
    self.n_layers = 2

    # Initializing model layers
    self.lstm = bnlstm.LSTM(cell_class=bnlstm.BNLSTMCell, input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.n_layers, max_length=152, batch_first=True, dropout=0.5)
    self.lstm_fc = nn.Linear(in_features=self.hidden_size, out_features=self.lstm_output_size)
    self.sigmoid = nn.Sigmoid()
    self.dropout = nn.Dropout(p=0.3)
    self.fc = nn.Linear(in_features=self.lstm_output_size, out_features=1)

    # Initializing dataset hiperparams
    self.window_size = 40
    self.num_test_person = 20

    # Initializing training hiperparams
    self.batch_size = 256
    self.base_lr = 4*1e-4
    
    # Initializing the loss function
    self.criterion = nn.BCELoss()



  def forward(self, x1, x2):
    # Output shape: (batch_size, lstm_output_size)
    x1, (hidden_states, cell_states) = self.lstm(x1)
    x1 = x1.permute(1, 0, 2)
    x1 = x1[:, -1, :]
    x1 = self.lstm_fc(x1)
    x1 = self.sigmoid(x1)

    x2, (hidden_states, cell_states) = self.lstm(x2)
    x2 = x2.permute(1, 0, 2)
    x2 = x2[:, -1, :]
    x2 = self.lstm_fc(x2)
    x2 = self.sigmoid(x2)

    x = torch.abs(x1 - x2)
    x = self.dropout(x)
    x = self.fc(x)
    x = self.sigmoid(x)

    return x

  

  def count_parameters(self):
    return sum(p.numel() for p in self.parameters() if p.requires_grad)

  

  def configure_optimizers(self):
    optimizer = optim.Adam(self.parameters(), lr=self.base_lr)
    return optimizer

  

  def training_step(self, batch, batch_idx):
    x1, x2, y = batch    
    y = torch.unsqueeze(y, dim=1)
    y_hat = self.forward(x1, x2)

    loss = self.criterion(y_hat, y)

    y_pred = y_hat > 0.5
    acc = accuracy_score(y.cpu().view(-1), y_pred.cpu().view(-1))

    # Logging to TensorBoard 
    self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
    self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    return {'loss': loss, 'y_hat': y_hat, 'y': y}



  def validation_step(self, batch, batch_idx):
    x1, x2, y = batch
    y = torch.unsqueeze(y, dim=1)    
    y_hat = self.forward(x1, x2)

    loss = self.criterion(y_hat, y)

    y_pred = y_hat > 0.5
    acc = accuracy_score(y.cpu().view(-1), y_pred.cpu().view(-1))

    # Logging to TensorBoard 
    self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
    self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)

    return {'loss': loss, 'y_hat': y_hat, 'y': y}

  

  def train_dataloader(self):
    # Called when training the model
    self.train_dataset = SiameseDataset(mode="train", window_size=self.window_size, num_test_person=self.num_test_person)
    return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)



  def val_dataloader(self):
    # Called when evaluating the model (for each "n" steps or "n" epochs)
    self.val_dataset = SiameseDataset(mode="val", window_size=self.window_size, num_test_person=self.num_test_person)
    return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)



if __name__ == "__main__":
  checkpoint_callback = ModelCheckpoint(
    monitor='val_acc',
    dirpath='model_params/siamese_network/',
    filename='val-siamese-{epoch:02d}-{val_acc:.2f}',
    save_top_k=5,
    mode='max',
  )

  trainer = Trainer(gpus=1, deterministic=True, max_epochs=50, callbacks=[checkpoint_callback], fast_dev_run=False)
  model = SiameseNetwork()
  trainer.fit(model)



  '''
  model_siam = SiameseNetwork(input_size=78, hidden_size=128, lstm_output_size=170, n_layers=2).to(device=torch.device('cuda:0'))
  x1 = torch.rand(2, 40, 78).to(device=torch.device('cuda:0'))
  x2 = torch.rand(2, 40, 78).to(device=torch.device('cuda:0'))

  outs = model_siam(x1, x2)

  print(outs)
  print(model_siam.count_parameters())
  '''



  '''
  model_lr = LogisticRegression(num_keypoints=78, num_features=2, num_classes=170)
  model_lstm = LSTM(input_size=78, hidden_size=128, num_classes=170, n_layers=2).to(device=torch.device('cuda:0'))

  x_lr = torch.rand(10, 78, 2)
  y_lr = model_lr(x_lr)
  print(y_lr.shape)


  x_lstm = torch.rand(10, 40, 78).to(device=torch.device('cuda:0'))
  y_lstm = model_lstm(x_lstm)
  print(y_lstm.shape)
  '''


