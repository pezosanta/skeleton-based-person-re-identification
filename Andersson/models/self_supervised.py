import torch
import torch.nn as nn

import models.bnlstm_module as bnlstm



class SelfSupervised(nn.Module):

  # sequence length: number of subalternating input pose sequences
  # input size: number of joints (20) in a skeleton x number of coordinates (2) per joint = 40
  # hidden size: the hidden size of the LSTM
  # num_classes: the output shape of the LSTM 
  # n_layers: number of stacked LSTM modules in the LSTM  
  def __init__(self, input_size, window_size, lstm_output_size, hidden_size=128, n_layers=2, dropout=0.0, supervised_mode=False):
    super(SelfSupervised, self).__init__()

    self.supervised_mode = supervised_mode
    self.input_size = input_size
    self.window_size = window_size

    self.bnlstm = bnlstm.LSTM(cell_class=bnlstm.BNLSTMCell, input_size=input_size, hidden_size=hidden_size, num_layers=n_layers, max_length=152, batch_first=True, dropout=dropout)
    self.fc = nn.Linear(in_features=hidden_size, out_features=lstm_output_size)
    self.decoder = nn.Linear(in_features=lstm_output_size, out_features=self.window_size*self.input_size)



  # Input (sequence) shape: (batch_size, sequence_length, input_size) e.g.: (64, 3, 40)
  def forward(self, sequences):
    # BNLSTM output shape: (sequence_length, batch_size, self.hidden_size)
    out, (hidden_states, cell_states) = self.bnlstm(sequences)
    out = out.permute(1, 0, 2)

    # Using only the output of the last sequence for prediction
    # Shape: (batch_size, self.hidden_size)
    out = out[:, -1, :]

    # Generating the output
    # Output shape: (batch_size, num_classes)
    out = self.fc(out)

    if self.supervised_mode:
        return out

    # Output shape: (batch_size, sequence_length, input_size)
    out = self.decoder(out)
    out = out.view(-1, self.window_size, self.input_size)

    return out

  

  def count_parameters(self):
    return sum(p.numel() for p in self.parameters() if p.requires_grad)