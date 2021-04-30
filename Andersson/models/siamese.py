import torch
import torch.nn as nn

from enum import Enum

from torch.nn.modules.activation import Sigmoid

from models.bnlstm import BNLSTM



class ModelMode(Enum):
    TRAINING = 0
    TEST_CREATING_LABELS = 1
    TEST_EVALUATING_LABELS = 2



class SiameseCrossentropy(nn.Module): 
    def __init__(self, model, model_output_size, latent_size, out_fc_size):
        super(SiameseCrossentropy, self).__init__()

        # Squared Eucledian distance, Cosine Similarity
        self.additional_features = 2
        
        self.model = nn.Sequential(
            model,
            nn.BatchNorm1d(num_features=model_output_size),
            nn.ReLU()
        )
        
        self.latent_fc = nn.Sequential(
            nn.Linear(in_features=model_output_size, out_features=latent_size-self.additional_features),
            nn.Sigmoid()
        )
        
        self.out_fc = nn.Sequential(
            nn.Linear(in_features=latent_size, out_features=out_fc_size),
            nn.BatchNorm1d(num_features=out_fc_size),
            nn.ReLU(),
            nn.Linear(in_features=out_fc_size, out_features=1),
            nn.Sigmoid()
        )

        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)



    # Input (sequence) shape: (batch_size, sequence_length, input_size) e.g.: (64, 3, 40)
    def forward(self, x1, x2=None, mode=ModelMode.TRAINING):
        if mode != ModelMode.TEST_EVALUATING_LABELS:
            # Output shape: (batch_size, lstm_output_size)
            x1 = self.model(x1)
            x1 = self.latent_fc(x1)

            # Returning the latent vector at testing phase when unique, single labels are created for each person
            if mode == ModelMode.TEST_CREATING_LABELS:
                return x1

            # Output shape: (batch_size, lstm_output_size)
            x2 = self.model(x2)
            x2 = self.latent_fc(x2)

        d1 = torch.abs(x1 - x2)                                             # Elementwise L1 distance
        d2 = torch.unsqueeze((torch.sum(((x1 - x2)**2), dim=1)), dim=1)     # Squared Eucledian distance
        d3 = torch.unsqueeze(self.cos(x1, x2), dim=1)                       # Cosine similarity
        x = torch.cat((d1, d2, d3), dim=1)

        x = self.out_fc(x)

        return x



    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)




from models.bnlstm import BNLSTM
#model, model_output_size, latent_size, out_fc_size
model = BNLSTM(input_size=78, lstm_output_size=160)
siamese = SiameseCrossentropy(model=model, model_output_size=160, latent_size=32, out_fc_size=16).cuda()
x = torch.zeros((2, 40, 78)).cuda()

print(siamese(x, -x, mode=ModelMode.TEST_CREATING_LABELS))