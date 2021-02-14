import torch
import torch.nn as nn
import bnlstm



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
