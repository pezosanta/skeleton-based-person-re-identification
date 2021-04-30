from datasets import SiameseDataset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from sklearn.metrics import accuracy_score




class SiameseNetwork(LightningModule):
  def __init__(self):
    super(SiameseNetwork, self).__init__()

    # Initializing model hiperparams
    self.input_size = 78
    self.hidden_size = 128
    self.lstm_output_size = 160 # 170
    self.latent_size = 32 # 170
    self.out_fc_size = 16 #32
    self.n_layers = 2

    model = BNLSTM(input_size=input_size, model_output_size=model_output_size, hidden_size=hidden_size, num_layers=n_layers, dropout=lstm_dropout)
    # Initializing model layers
    self.lstm = bnlstm.LSTM(cell_class=bnlstm.BNLSTMCell, input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.n_layers, max_length=152, batch_first=True, dropout=0.5)
    self.lstm_fc = nn.Sequential(
                nn.Linear(in_features=self.hidden_size, out_features=self.lstm_output_size),
                nn.BatchNorm1d(num_features=self.lstm_output_size),
                nn.ReLU(),
                nn.Linear(in_features=self.lstm_output_size, out_features=self.latent_size-2),
                nn.BatchNorm1d(num_features=self.latent_size-2),
                nn.Sigmoid()
    )

    self.sigmoid = nn.Sigmoid()
    self.dropout = nn.Dropout(p=0.5)

    self.out_fc = nn.Sequential(
                nn.Linear(in_features=self.latent_size, out_features=self.out_fc_size),
                nn.BatchNorm1d(num_features=self.out_fc_size),
                nn.ReLU(),
                nn.Linear(in_features=self.out_fc_size, out_features=1)
    )
    self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)

    # Initializing dataset hiperparams
    self.window_size = 40
    self.num_test_person = 20

    # Initializing training hiperparams
    self.batch_size = 512
    self.base_lr = 3e-3
    self.weight_decay = 0.01
    
    # Initializing the loss function
    self.criterion = nn.BCELoss()



  def forward(self, x1, x2=None, mode='all'):
    if mode != 'test_labels':
      # Output shape: (batch_size, lstm_output_size)
      x1, (hidden_states, cell_states) = self.lstm(x1)
      x1 = x1.permute(1, 0, 2)
      x1 = x1[:, -1, :]
      x1 = self.lstm_fc(x1)
      x1 = self.sigmoid(x1)

      if mode == 'create_labels':
        return x1

      x2, (hidden_states, cell_states) = self.lstm(x2)
      x2 = x2.permute(1, 0, 2)
      x2 = x2[:, -1, :]
      x2 = self.lstm_fc(x2)
      x2 = self.sigmoid(x2)

    d1 = torch.abs(x1 - x2)                                         # Elementwise L1 distance
    d2 = torch.unsqueeze((torch.sum(((x1 - x2)**2), dim=1)), dim=1) # Squared Eucledian distance
    d3 = torch.unsqueeze(self.cos(x1, x2), dim=1)                   # Cosine similarity
    x = torch.cat((d1, d2, d3), dim=1)

    x = self.dropout(x)
    x = self.out_fc(x)
    x = self.sigmoid(x)

    return x

  

  def count_parameters(self):
    return sum(p.numel() for p in self.parameters() if p.requires_grad)

  

  def configure_optimizers(self):
    optimizer = optim.Adam(self.parameters(), lr=self.base_lr, weight_decay=self.weight_decay)
    #lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=5, T_mult=2, eta_min=1e-5)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[5, 10, 15, 25], gamma=0.33)
    return [optimizer], [lr_scheduler]

  

  def training_step(self, batch, batch_idx):
    x1, x2, y = batch    
    y = torch.unsqueeze(y, dim=1)
    y_hat = self.forward(x1, x2)

    loss = self.criterion(y_hat, y)

    y_pred = (y_hat > 0.5).type(torch.float32)
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

    y_pred = (y_hat > 0.5).type(torch.float32)
    acc = accuracy_score(y.cpu().view(-1), y_pred.cpu().view(-1))

    # Logging to TensorBoard 
    self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=False, logger=True)
    self.log('val_acc', acc, on_step=True, on_epoch=True, prog_bar=False, logger=True)

    return {'loss': loss, 'y_hat': y_hat, 'y': y}

  

  def train_dataloader(self):
    # Called when training the model
    self.train_dataset = SiameseDataset(dataset="train", window_size=self.window_size, num_test_person=self.num_test_person)
    return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)



  def val_dataloader(self):
    # Called when evaluating the model (for each "n" steps or "n" epochs)
    self.val_dataset = SiameseDataset(dataset="val", window_size=self.window_size, num_test_person=self.num_test_person)
    return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)



if __name__ == "__main__":
  
  checkpoint_callback = ModelCheckpoint(
    monitor='val_acc',
    dirpath='model_params/siamese_network/version_6_bs_512_baselr_3e-3_multisteplr_milestones_5-10-15-25_gamma_0.33_lstoutsize_32/',
    filename='version-6-siamese-{epoch:02d}-{val_acc_epoch:.4f}',
    save_top_k=35,
    mode='max',
  )

  lr_monitor = LearningRateMonitor(logging_interval='epoch')

  trainer = Trainer(gpus=1, deterministic=True, max_epochs=35, callbacks=[checkpoint_callback, lr_monitor], fast_dev_run=False)
  model = SiameseNetwork()
  trainer.fit(model)
  



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


