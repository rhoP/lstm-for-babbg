import torch
from torch import nn




class bbgLSTM(nn.Module):
    """ This is the first option I thought of. Say you have k inputs and you need k outputs. 
    Just run the lstm k times in sequence and discard the first k-1 outputs.
     The kth output is the first one after the input sequence and then continue the lstm sequence for another k-1 steps
    """
    def __init__(self, input_dim):
        super(bbgLSTM, self).__init__()

        self._input_dim = input_dim    #The size here is the number of features, the actual input will be k\times features, change it if necessary
        self._hidden_dim = input_dim   #has to be the same size of the input
        self._num_layers = 1           #Just set this to the default value of 1

        self.lstm = nn.LSTM(self._input_dim, self._hidden_dim, self._num_layers)


    def init_hidden(self, batch_size):
        return(torch.randn(batch_size, self._hidden_dim),
						torch.randn(batch_size, self._hidden_dim))
    
    def forward(self, batch):
        """Batch is probably going to look like k x batch_size x features and so just pass k as steps"""
        hidden, state = self.init_hidden(batch.shape[-2])
        outputs = []
        output = [] # just a placeholder
        if batch.ndim<3: #Unnecessary but I like to be extra
            raise Exception("Danger, Danger Will Robinson")
        elif batch.shape[0] > 1:
            steps = range(batch.shape[0] - 1)
        else:
            raise Exception("ya done effed up, kid")
        

        for step in steps:
            output, (hidden, c) = self.lstm(batch[step, :, :], (hidden, state))
        
        outputs.append(output)
        
        for step in steps:
            output, (hidden, c) = self.lstm(output, (hidden, state))
            outputs.append(output)
        
        return outputs

