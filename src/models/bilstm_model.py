import random
random.seed(22)

import torch 
torch.manual_seed(22)
import torch.nn as nn 
import torch.nn.functional as F 
import time 

use_gpus = torch.cuda.is_available()
if use_gpus:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

class BILSTM(nn.Module):
    def __init__(self, args, vocab_size):
        super(BILSTM, self).__init__()

        self.hidden_dim = args.hidden_dim 
        self.batch_size = args.batch_size 
        self.dropout = args.dropout 
        self.embedding_dim = args.embedding_dim

        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim, bidirectional=True)

        self.fc = nn.Linear(self.hidden_dim * 2, 2)
    
    def forward(self, x):
        
        x = self.embedding(x).view(x.shape[0], self.batch_size, -1)

        h0 = torch.zeros(2, self.batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(2, self.batch_size, self.hidden_dim).to(device)

        rnn, self.hidden = self.lstm(x, (h0, c0))

        # only use last states as input to dense layer
        y = self.fc(rnn[-1])

        log_softmax_y = F.log_softmax(y, dim=1)

        return log_softmax_y




