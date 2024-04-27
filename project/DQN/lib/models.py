import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time



# class DQNConv1D(nn.Module):
#     def __init__(self, shape, actions_n):
#         """捲積網路的參數詳解
#             in_channels = 特徵的數量
#             out_channels 出來的特徵數量(如果有100張,可以想像就是有100張小圖)
#             kernel_size (濾波器的大小)
#             stride = (移動步長)
#             padding =補0            
            
#             # 這邊指的是input 是指輸入的特徵,口訣就是批次大小,通道數,數據長度
            
#             input_data 是三维的，因为卷积层（nn.Conv1d）的输入需要具有特定的形状，以便正确地进行卷积操作。这三个维度分别代表：

#             批次大小（Batch Size）：

#             代表一次输入到网络中的样本数量。通过一次处理多个样本（一个批次），网络可以加速训练，并且可以得到梯度的更稳定估计。
            
#             通道数（Channels）：

#             对于图像，通道通常代表颜色通道（例如，RGB）。在其他类型的数据中，通道可以代表不同类型的输入特征。
#             在一维卷积中，通道数可能对应于输入数据的不同特征。
            
#             数据长度（Data Length）：

#             这是输入数据的实际长度。对于图像，这会是图像的宽度和高度。对于一维数据，如时间序列，这会是序列的长度。
#         Args:
#             shape (_type_): _description_
#             actions_n (_type_): _description_
#         """
#         super(DQNConv1D, self).__init__()

#         self.conv = nn.Sequential(
#             nn.Conv1d(shape[0], 128, kernel_size=5),
#             nn.ReLU(),
#             nn.Conv1d(128, 128, 5),
#             nn.ReLU(),
#         )

#         out_size = self._get_conv_out(shape)

#         self.fc_val = nn.Sequential(
#             nn.Linear(out_size, 512),
#             nn.ReLU(),
#             nn.Linear(512, 1)
#         )

#         self.fc_adv = nn.Sequential(
#             nn.Linear(out_size, 512),
#             nn.ReLU(),
#             nn.Linear(512, actions_n)
#         )

#     def forward(self, x):
#         # 要丟進去一般網絡之前要先flatten
#         conv_out = self.conv(x).view(x.size()[0], -1) #執行flatten的動作


#         val = self.fc_val(conv_out)
#         adv = self.fc_adv(conv_out)
#         return val + adv - adv.mean(dim=1, keepdim=True)
    
#     def _get_conv_out(self, shape):
#         """用來計算flatten後的參數個數

#         Args:
#             shape (_type_): _description_

#         Returns:
#             _type_: _description_
#         """
#         o = self.conv(torch.zeros(1, *shape))
#         return int(np.prod(o.size()))

    


# 還可以加入dropout 和 批次正則
class DQNConv1D_Large(nn.Module):
    def __init__(self, shape, actions_n):
        super(DQNConv1D_Large, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(shape[0], 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # Max pooling layer added here
            
            nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # Another max pooling layer
            
            nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),  # And another
            
            nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)  # Final max pooling layer
        )

        out_size = self._get_conv_out(shape)

        self.fc_val = nn.Sequential(
            nn.Linear(out_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )

        self.fc_adv = nn.Sequential(
            nn.Linear(out_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, actions_n)
        )

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)  # Flatten the tensor after conv layers

        val = self.fc_val(conv_out)
        adv = self.fc_adv(conv_out)
        return val + adv - adv.mean(dim=1, keepdim=True)

    def _get_conv_out(self, shape):
        with torch.no_grad():  # To ensure it doesn't track history in autograd
            o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))





# import torch
# import torch.nn as nn
# import torch.optim as optim

# class StockTradingModel(nn.Module):
#     def __init__(self, input_channels, hidden_dim, num_layers, output_dim):
#         super(StockTradingModel, self).__init__()
#         self.num_layers = num_layers
#         self.hidden_dim = hidden_dim
        
#         # Convolutional Layer
#         self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1)
#         self.relu = nn.ReLU()
        
#         # LSTM Layer
#         self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        
#         # Fully Connected Layer
#         self.fc = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x):
#         # Convolutional operations
#         x = x.permute(0, 2, 1)  # Adjusting dimensions for Conv1D (Batch, Channels, Length)
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = x.permute(0, 2, 1)  # Adjust back dimensions for LSTM
        
#         # LSTM operations
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)  # Initial hidden state
#         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)  # Initial cell state
#         x, _ = self.lstm(x, (h0, c0))
        
#         # Fully connected layer
#         x = self.fc(x[:, -1, :])  # We only use the last LSTM output
#         return x

# # Example settings
# input_channels = 1  # Number of features per time step
# hidden_dim = 64
# num_layers = 2
# output_dim = 1  # Output dimension

# # Creating the model
# model = StockTradingModel(input_channels, hidden_dim, num_layers, output_dim)
# print(model)
