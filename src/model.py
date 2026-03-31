from torch import nn
import torch
import config

class ReviewAnalyzeModel(nn.Module):
    def __init__(self,vocab_size,padding_index):
        super().__init__()
        self.padding_index = padding_index
        self.embedding = nn.Embedding(vocab_size,config.EMBEDDING_DIM,padding_idx = self.padding_index)
        self.lstm = nn.LSTM(input_size = config.EMBEDDING_DIM, hidden_size = config.HIDDEN_SIZE,
                            batch_first = True)
        self.linear = nn.Linear (config.HIDDEN_SIZE,1)

    def forward(self , x:torch.Tensor):
        # x.shape:[batch_size,seq_len]
        embed = self.embedding(x)
        #embed.shape: [batch_size,seq_len,embedding_dim]
        output,(hn,cn) = self.lstm(embed)
        # output.shape :[batch_size,seq_len,hidden_size]
        #取出每个样本最后的隐藏状态 真实的 排除puk 填充
        batch_indexs = torch.arange(0,output.shape[0])
        lengths = (x != self.padding_index ).sum(dim=1)
        last_hidden = output[batch_indexs,lengths-1]
        output = self.linear(last_hidden).squeeze(-1)
        return output