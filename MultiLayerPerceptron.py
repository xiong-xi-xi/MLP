import numpy as np
from utils import *

class Single_Layer():
    def __init__(self, input_dim, output_dim, init_method, weight_scale, activate_func, regularization, l_lambda):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.w = init_w(input_dim, output_dim, init_method, weight_scale)
        self.b = init_b(output_dim, weight_scale)
        self.regularization = regularization
        self.l_lambda = l_lambda

        self.grad_z = None
        self.pre_output = None
        self.dw = None
        self.db = None

        if activate_func == "Relu":
            self.activation = Relu()
        elif activate_func == "LeakyRelu":
            self.activation = LeakyRelu()
        else:
            self.activation = Sigmoid()

    def forward(self, x):
        '''
        前向传播过程
        :param x: 输入
        :return: 返回经过激活函数后的值
        '''

        self.pre_output = x
        y = np.dot(x, self.w) + self.b
        z = self.activation(y)

        self.grad_z = self.activation.derivative(y)

        return z

    def backward(self, d_error_out):
        '''
        反向传播过程 error->out->net->w,b
        :param d_error_out: error对out求导
        :return: 返回给上一层的error对out的求导
        '''

        d_out_net = self.grad_z  # d_out_net代表out对net的求导结果
        d_net_w = self.pre_output # d_net_w代表net对w的求导结果

        d_error_net = np.multiply(d_out_net, d_error_out) # d_error_net代表error对net的求导结果，这里应用了链式法则
        d_error_w = np.dot(d_net_w.T, d_error_net) # d_error_w代表error对w的求导结果，这里应用了链式法则

        self.dw = d_error_w

        d_error_b = d_error_net.sum(axis=0)
        self.db = d_error_b

        d_error_out_ = np.dot(d_error_net, self.w.T) # 给下一层的error对out的求导的结果为上一层的加权和

        return d_error_out_

    def update(self, lr):

        if self.regularization == 1: # L1正则化
            self.dw += self.l_lambda * np.mean(np.abs(self.w))
        elif self.regularization == 2: # L2正则化
            self.dw += self.l_lambda * np.sqrt(np.mean(np.square(self.w)))
        else: # 不正则化
            self.dw = self.dw

        self.w -= lr * self.dw
        self.b -= lr * self.db

        return None

class Multi_Layer():
    def __init__(self, layer_dim_list, init_method, weight_scale, activate_func, loss_func, lr, regularization, l_lambda):
        self.layer_list = []
        self.lr = lr
        self.regularization = regularization

        for i in range(len(layer_dim_list)-1):
            input_dim = layer_dim_list[i]
            output_dim = layer_dim_list[i+1]
            self.layer_list.append(Single_Layer(input_dim, output_dim, init_method, weight_scale, activate_func, regularization, l_lambda))
        self.layer_num = len(self.layer_list)

        if loss_func == 'MSE':
            self.loss_func = MeanSquaredLoss()
        else:
            self.loss_func = CrossEntropyLoss()

    def forward(self, input, target):
        input = input.reshape(input.shape[0], -1)
        out = input
        for layer in self.layer_list:
            out = layer.forward(input)
            input = out

        loss = self.loss_func(out, target)

        return out, loss

    def backward(self, predict, target):
        d_error_out = self.loss_func.derivative(predict, target)

        for idx in range(self.layer_num-1, -1, -1): # 从后往前进行梯度更新
            d_error_out = self.layer_list[idx].backward(d_error_out)

        for layer in self.layer_list: # SGD更新每层的权重
            layer.update(self.lr)
