
import torch
import torch.nn as nn



class LSTM(nn.Module):

  # sequence length: number of subalternating input pose sequences
  # input size: number of joints (20/39) in a skeleton x number of coordinates (2) per joint = 40/78
  # hidden size: the hidden size of the LSTM
  # num_classes: the output shape of the LSTM 
  # n_layers: number of stacked LSTM modules in the LSTM  
  def __init__(self, input_size, lstm_output_size, hidden_size=256, n_layers=2, dropout=0.5):
    super(LSTM, self).__init__()

    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    self.hidden_size = hidden_size
    self.n_layers = n_layers

    self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=n_layers, batch_first=True, dropout=0.3)
    self.fc = nn.Linear(in_features=hidden_size, out_features=lstm_output_size)
    self.dropout = nn.Dropout(p=dropout)



  def _reset_hidden_state(self, batch_size):
    hidden = (torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device=self.device),
              torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device=self.device))
    
    return hidden



  # Input (sequence) shape: (batch_size, sequence_length, input_size) e.g.: (64, 40, 78)
  def forward(self, sequences):
    # Reseting hidden states in every iteration
    # Hidden shape: (self.n_layers, batch_size, self.hidden_size)
    hidden_states = self._reset_hidden_state(batch_size=sequences.shape[0])

    # LSTM output shape: (batch_size, sequence_length, self.hidden_size)
    out, hidden_states = self.lstm(sequences, hidden_states)

    # Using only the output of the last sequence for prediction
    # Shape: (batch_size, self.hidden_size)
    out = out[:, -1, :]

    # Dropout for regularization
    out = self.dropout(out)

    # Generating the output
    # Output shape: (batch_size, num_classes)
    out = self.fc(out)

    return out

  

  def count_parameters(self):
    return sum(p.numel() for p in self.parameters() if p.requires_grad)

