import numpy as np
import random

np.random.seed(2020)
random.seed(2020)

'''hyperpara example

init_method = 'Normal' # Xavier
weight_scale = 0.01
activate_func = 'Relu' # Sigmoid
loss_func = 'CrossEntropyLoss' # MSE
lr = 0.2 # 1
epoch = 50
batch_size = 128
regularization = 2 # 0：不进行正则化, 1: L1正则化, 2：L2正则化
l_lambda = 0.001
'''

epoch = 50
batch_size = 128

dict0 = dict(layer_dim = [784, 800, 10],
             init_method = 'Normal', # Xavier
             weight_scale = 0.01,
             activate_func = 'Relu',
             loss_func = 'CrossEntropyLoss',
             lr = 0.2,
             regularization = 0, # 0：不进行正则化, 1: L1正则化, 2：L2正则化
             l_lambda = 0.001
             )

# 基于dict0改损失函数
dict1 = dict(layer_dim = [784, 800, 10],
             init_method = 'Normal',
             weight_scale = 0.01,
             activate_func = 'Relu',
             loss_func = 'MSE',
             lr = 1,
             regularization = 0, # 0：不进行正则化, 1: L1正则化, 2：L2正则化
             l_lambda = 0.001
             )

# 基于dict1改lr
dict2 = dict(layer_dim = [784, 800, 10],
             init_method='Normal',
             weight_scale = 0.01,
             activate_func = 'Relu',
             loss_func = 'MSE',
             lr = 0.8,
             regularization = 0, # 0：不进行正则化, 1: L1正则化, 2：L2正则化
             l_lambda = 0.001
             )

# 基于dict1改regularization
dict3 = dict(layer_dim = [784, 800, 10],
             init_method='Normal',
             weight_scale = 0.01,
             activate_func = 'Relu',
             loss_func = 'MSE',
             lr = 1,
             regularization = 1, # 0：不进行正则化, 1: L1正则化, 2：L2正则化
             l_lambda = 0.001
             )

# 基于dict1改regularization
dict4 = dict(layer_dim = [784, 800, 10],
             init_method='Normal',
             weight_scale = 0.01,
             activate_func = 'Relu',
             loss_func = 'MSE',
             lr = 1,
             regularization = 2, # 0：不进行正则化, 1: L1正则化, 2：L2正则化
             l_lambda = 0.001
             )

# 基于dict1改init
dict5 = dict(layer_dim = [784, 800, 10],
             init_method='Xavier',
             weight_scale = 0.01,
             activate_func = 'Relu',
             loss_func = 'MSE',
             lr = 1,
             regularization = 0, # 0：不进行正则化, 1: L1正则化, 2：L2正则化
             l_lambda = 0.001
             )

# 基于dict5改activate
dict6 = dict(layer_dim = [784, 800, 10],
             init_method='Normal',
             weight_scale = 0.01,
             activate_func = 'Sigmoid',
             loss_func = 'MSE',
             lr = 1,
             regularization = 0, # 0：不进行正则化, 1: L1正则化, 2：L2正则化
             l_lambda = 0.001
             )

hyper_dict = [
dict0,
dict1,
dict2,
dict3,
dict4,
dict5,
dict6,
]