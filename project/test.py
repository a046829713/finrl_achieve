# import torch
# import time

# def conv1d_custom(input, weight, stride=1, padding=0):
#     # 計算輸入和輸出的大小
#     batch_size, in_channels, width = input.shape
#     out_channels, _, kernel_size = weight.shape
#     output_width = (width - kernel_size + 2 * padding) // stride + 1

#     # 應用填充
#     if padding > 0:
#         input_padded = torch.nn.functional.pad(input, (padding, padding))
#     else:
#         input_padded = input

#     # 初始化輸出
#     output = torch.zeros((batch_size, out_channels, output_width))

#     # 執行卷積操作
#     for i in range(output_width):
#         # 計算輸入切片的範圍
#         start = i * stride
#         end = start + kernel_size
#         # 提取輸入切片並與卷積核進行乘法運算
#         input_slice = input_padded[:, :, start:end]
#         output[:, :, i] = torch.sum(input_slice.unsqueeze(1) * weight, dim=(2, 3))

    
#     return output

# # 測試自定義的卷積函數
# batch_size = 1
# in_channels = 3
# out_channels = 2
# kernel_size = 3
# stride = 1
# padding = 1
# length = 50

# # 創建隨機輸入和卷積核
# input_data = torch.randn(batch_size, in_channels, length)
# weight = torch.randn(out_channels, in_channels, kernel_size)

# # 使用自定義函數進行卷積
# output = conv1d_custom(input_data, weight, stride=stride, padding=padding)





# import torch
# import torch.nn as nn

# # 定義原始自定義函數的參數
# batch_size = 1
# in_channels = 3
# out_channels = 2
# kernel_size = 3
# stride = 1
# padding = 1
# length = 50

# # 創建隨機輸入和卷積核
# input_data = torch.randn(batch_size, in_channels, length)
# weight = torch.randn(out_channels, in_channels, kernel_size)

# # 使用 nn.Conv1d 重寫自定義函數
# conv1d_layer = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
#                          kernel_size=kernel_size, stride=stride, padding=padding)

# # 將自定義函數的權重賦給 nn.Conv1d
# with torch.no_grad():
#     conv1d_layer.weight = nn.Parameter(weight)

# # 使用 nn.Conv1d 進行卷積
# output_nn = conv1d_layer(input_data)

# output_nn.shape, output_nn

print('LastPortfolioAdjustmentTime'.lower())
# a = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'AVAXUSDT', 'TRBUSDT', 'ETCUSDT', 'BNBUSDT', 'INJUSDT', 'LTCUSDT', 'BCHUSDT', 'MKRUSDT', 'AAVEUSDT', 'SSVUSDT', 'EGLDUSDT', 'COMPUSDT', 'XMRUSDT', 'NMRUSDT', 'KSMUSDT', 'ZECUSDT', 'YFIUSDT', 'GMXUSDT', 'QNTUSDT', 'DASHUSDT', 'FOOTBALLUSDT', 'BTCDOMUSDT', 'DEFIUSDT']
# b = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'AVAXUSDT', 'TRBUSDT']


# 總數不能超過11
# 