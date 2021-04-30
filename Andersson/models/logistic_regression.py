import torch.nn as nn


class LogisticRegression(nn.Module):
  def __init__(self, num_keypoints, num_features, output_size):
    super(LogisticRegression, self).__init__()

    self.linear = nn.Linear(in_features=num_keypoints*num_features, out_features=output_size)



  def forward(self, x):
    x = x.view(x.size(0), -1)
    x = self.linear(x)

    return x
  


  def count_parameters(self):
    return sum(p.numel() for p in self.parameters() if p.requires_grad)