import numpy as np
from abc import abstractmethod

def Accuracy(predict, label):
    out_p = np.argmax(predict, axis=1)
    label_p = np.argmax(label, axis=1)

    return np.sum(out_p == label_p) / out_p.shape[0]

def init_w(input_dim, output_dim, init_method, weight_scale):
    if init_method == 'Normal':
        w = weight_scale * np.random.randn(input_dim, output_dim)
    if init_method == 'Xavier':
        w = np.sqrt(2 / (input_dim + output_dim)) * np.random.randn(input_dim, output_dim)
    return w

def init_b(output_dim, weight_scale):
    return weight_scale * np.random.randn(1, output_dim)

# 激活函数模块
class ActivateFunction():
    @abstractmethod
    def __call__(self, y):
        return None
    @abstractmethod
    def derivative(self, y):
        # derivative with respect to y_pred
        return None

class Sigmoid(ActivateFunction):
    def __call__(self, y):
        return 1.0 / (1.0 + np.exp(-y))
    def derivative(self, y):
        z = 1.0 / (1.0 + np.exp(-y))
        return z * (1 - z)

class Relu(ActivateFunction):
    def __call__(self, y):
        return np.maximum(0, y)
    def derivative(self, y):
        y_ = np.ones_like(y)
        y_[(y<=0)] = 0
        return y_

class LeakyRelu(ActivateFunction):
    def __call__(self, y):
        return np.where(y > 0, y, 0.01)
    def derivative(self, y):
        y_ = np.copy(y)
        y_[y > 0] = 1
        y_[y <= 0] = 0
        return y_

class Softmax(ActivateFunction):
    def __call__(self, y):
        y_max = np.max(y, axis=1).reshape(y.shape[0], 1)
        y = y - y_max
        exp = np.exp(y)
        sumexp = np.sum(exp, axis=1).reshape(np.sum(exp, axis=1).shape[0], 1)
        return exp / sumexp

    def derivative(self, y):
        d_softmax = []
        for i in range(y.shape[0]):
            d_softmax.append(np.diag(y[i]) - np.dot(y[i], y[i].T))
        d_softmax = np.array(d_softmax)

        return d_softmax

# 损失函数模块
class LossFunction():
    @abstractmethod
    def __call__(self, y_pred, y_label):
        return None
    @abstractmethod
    def derivative(self, y_pred, y_label):
        return None

class MeanSquaredLoss(LossFunction):
    def __init__(self):
        self.softmax = Softmax()

    def __call__(self, y_pred, y_label):
        y_pred = self.softmax(y_pred)
        return np.mean(np.sum(np.square(y_pred - y_label), axis=1), axis=0)

    def derivative(self, y_pred, y_label):
        y_pred = self.softmax(y_pred)
        d_softmax = self.softmax.derivative(y_pred)
        dx = [np.dot(d_softmax[i], (y_pred[i] - y_label[i])) for i in range(y_pred.shape[0])]
        dx = 2 * np.array(dx) / y_pred.shape[0]
        return dx

class CrossEntropyLoss(LossFunction):
    def __init__(self):
        self.softmax = Softmax()

    def __call__(self, y_pred, y_label):
        y_pred = self.softmax(y_pred)
        return -np.sum(y_label * np.log(y_pred + 1e-7)) / y_pred.shape[0]

    def derivative(self, y_pred, y_label):
        y_pred = self.softmax(y_pred)
        dx = (y_pred - y_label) / y_pred.shape[0]

        return dx






