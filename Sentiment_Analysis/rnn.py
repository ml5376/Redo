import torch
import torch.nn as nn
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
    PackedSequence
)
import torch.nn.functional as F



class RNNClassifier(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, use_lstm=True, **additional_kwargs):
        """
        Inputs:
            num_embeddings: size of the vocabulary
            embedding_dim: size of an embedding vector
            hidden_size: hidden_size of the rnn layer
            use_lstm: use LSTM if True, vanilla RNN if false, default=True
        """
        super().__init__()

        # Change this if you edit arguments
        self.hparams = {
            'num_embeddings': num_embeddings,
            'embedding_dim': embedding_dim,
            'hidden_size': hidden_size,
            'use_lstm': use_lstm,
            **additional_kwargs
        }
        if use_lstm==False:
            self.e=nn.Embedding(num_embeddings,embedding_dim)
            self.encoder=nn.RNN(embedding_dim,hidden_size,nonlinearity="relu")
            for p in self.rnn.parameters():
                nn.init.normal(p,mean=0.0,std=0.001)
            self.decoder=nn.Linear(1*hidden_size, 16)
            self.sig=nn.Sigmoid()

        elif use_lstm==True:
            self.e=nn.Embedding(num_embeddings,embedding_dim)
            self.encoder=nn.LSTM(embedding_dim,hidden_size,num_layers=2,dropout=0.5)
            self.decoder=nn.Linear(2*hidden_size,1) #output a probability 
            self.drop=nn.Dropout(0.5)
            self.sig=nn.Sigmoid()


    def forward(self, sequence, lengths=None):
        """
            Inputs
                sequence: A long tensor of size (seq_len, batch_size)
                lengths: A long tensor of size batch_size, represents the actual
                    sequence length of each element in the batch. If None, sequence
                    lengths are identical.
            Outputs:
                output: A 1-D tensor of size (batch_size,) represents the probabilities of being
                    positive, i.e. in range (0, 1)
        """
        output = None

        ########################################################################
        # TODO: Apply the forward pass of your network                         #
        # hint: Don't forget to use pack_padded_sequence if lenghts is not None#
        # pack_padded_sequence should be applied to the embedding outputs      #
        ########################################################################

        output = self.drop(self.e(sequence))


        if lengths is not None:

           output=pack_padded_sequence(output,lengths=lengths,batch_first=False,enforce_sorted=False)
        #    output=pack_padded_sequence(output,lengths=lengths,enforce_sorted=False)
        
    
        outputs, (hidden,cell) = self.encoder(output) # output, (h, c)


        hidden = torch.cat((hidden[-2], hidden[-1]),dim=1)
     
       
       
        outputs=self.drop(hidden)
        output=self.decoder(outputs)
        output=torch.squeeze(output,1)
        output=self.sig(output)

    
        return output
