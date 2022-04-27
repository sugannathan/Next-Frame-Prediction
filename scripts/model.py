import torch
import torch.nn as nn

from encoder import Embedded_Encoder
from decoder import Embedded_Decoder
from conv_lstm import ConvLSTM

class Model(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.embd_model = Embedded_Encoder()
        self.conv_lstm_1 = ConvLSTM(input_dim= 64, hidden_dim = 64, kernel_size = (3, 3), num_layers= 2)
        self.conv_lstm_2 = ConvLSTM(input_dim= 128, hidden_dim = 128, kernel_size = (3, 3), num_layers= 2)
        self.conv_lstm_3 = ConvLSTM(input_dim= 256, hidden_dim = 256, kernel_size = (3, 3), num_layers= 2)
        self.decoder = Embedded_Decoder()


    def forward(self, x):

        embds = self.embd_model(x)
        batch_size, seq_len = x.shape[0:2]
        
        lstm_1 = self.conv_lstm_1(embds[0]) #[Batch_size, t, 64, 16, 16]
        lstm_2 = self.conv_lstm_2(embds[1]) #[Batch_size, t, 128, 8, 8]
        lstm_3 = self.conv_lstm_3(embds[2]) #[Batch_size, t, 256, 4, 4]
        
        lstm_outputs = [lstm_3, lstm_2, lstm_1]

        out = self.decoder(lstm_outputs, embds, batch_size, seq_len)
        return out