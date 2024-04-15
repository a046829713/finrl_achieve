import torch
import sys

data = torch.load('DQN\Meta\BTCUSDT-300B-30K.pt')
print(sys.getsizeof(data))