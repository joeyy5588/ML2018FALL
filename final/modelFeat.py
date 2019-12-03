
import torch.nn as nn
import torch
import torch.nn.functional as F
from parameters import *


class ModelFeat(nn.Module):
    def __init__(self):
        super().__init__()
        # self.encoder = nn.LSTM(input_size = FEATURE_DIM, hidden_size = HIDDEN_SIZE, num_layers = NUM_LAYERS, dropout = LSTM_DROPOUT, batch_first = True)
        
        self.F = nn.Sequential(
              nn.Linear(FEATURE_DIM, D1_DIM),
              nn.ReLU(),
              nn.Dropout(),
              nn.Linear(D1_DIM, D2_DIM),
              nn.ReLU(),
        )
        
        self.S = nn.Sequential()
      
        self.classifier = nn.Sequential(
              nn.Linear(D2_DIM * 2, D2_DIM // 4),
              nn.ReLU(),
              nn.Linear(D2_DIM // 4, 1),
              nn.Sigmoid()
       )

    def forward(self, features, caption):
        f_output = self.F(features)
        s_output = self.S(caption)
        #print(f_output.shape,s_output.shape)
        c_input = torch.cat((f_output, s_output), dim = 1)
        c_output = self.classifier(c_input)
        return c_output