import math
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import time

# 可能的構想
# 是否要將部位損益等資訊從這邊移除?


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerDuelingModel(nn.Module):
    def __init__(self,
                 d_model: int,
                 nhead: int,
                 d_hid: int,
                 nlayers: int,
                 num_actions: int,
                 hidden_size: int,
                 seq_dim: int = 300,
                 dropout: float = 0.5,
                 batch_first=True):
        super().__init__()
        self.batch_first = batch_first
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)
        
        encoder_layers = TransformerEncoderLayer(
            hidden_size, nhead, d_hid, dropout, batch_first=self.batch_first)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        # 狀態值網絡
        self.fc_val = nn.Sequential(
            nn.Linear(seq_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        # 優勢網絡
        self.fc_adv = nn.Sequential(
            # input_dim = 6,hidden_size = 1024
            nn.Linear(seq_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_actions)
        )

        self.linear = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        # 將資料映射
        self.embedding = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )


    def forward(self, src: Tensor) -> Tensor:
        """
        Arguments:
            src: Tensor, shape ``[batch_size, seq_len, d_model]``

        Returns:
            output Tensor of shape ``[batch_size, num_actions]``
            
        """
        src = self.embedding(src)
        
        # src = torch.Size([1, 300, 6])
        if self.batch_first:
            src = self.pos_encoder(src.transpose(0, 1))
        else:
            src = self.pos_encoder(src)

        if self.batch_first:
            output = self.transformer_encoder(src.transpose(0, 1))
            
        else:
            output = self.transformer_encoder(src)

        # output = torch.Size([1, 300, 6])
        x = self.linear(output)
        x = x.view(x.size(0), -1)

        value = self.fc_val(x)
        # 狀態值和優勢值
        advantage = self.fc_adv(x)

        # 計算最終的Q值
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values