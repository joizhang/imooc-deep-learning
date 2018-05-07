import random

import numpy as np


class NetWork:
    def __init__(self, sizes):
        # 网络层数
        self.num_layers = len(sizes)
        # 每层神经元的个数
        self.sizes = sizes

        self.biases = None
        self.weights = None
        self.default_weight_initializer()

    def default_weight_initializer(self):
        # 初始化每层的偏置
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        # 初始化每层的权重
        self.weights = [np.random.randn(y, x) / np.sqrt(x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feed_forward(self, a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def gd(self, training_data, epochs, eta):
        """
        梯度下降
        :param eta:
        :param training_data: 训练数据集
        :param epochs: 需要训练的元素
        :return:
        """
        # 开始训练，循环每一个epochs
        for j in range(epochs):
            # 洗牌，打乱训练数据
            random.shuffle(training_data)

            # 保存每一层的偏导数
            nable_b = [np.zeros(b.shape) for b in self.biases]
            nable_w = [np.zeros(w.shape) for w in self.weights]

            # 训练每一个数据
            for x, y in training_data:
                delta_nable_b, delta_nable_w = self.back_prop(x, y)
                # 保存一次训练网络中每层的偏导数
                nable_b = [nb + dnb for nb, dnb in zip(nable_b, delta_nable_b)]
                nable_w = [nw + dnw for nw, dnw in zip(nable_w, delta_nable_b)]

            # 更新权重和偏置 w 下标n+1 = w 下标n - eta * nw
            self.weights = [w - eta * nw for w, nw in zip(self.weights, nable_w)]
            self.biases = [b - eta * nb for b, nb in zip(self.biases, nable_b)]
            print("Epoch %d complete" % j)

    def sgd(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """
        随机梯度下降
        :param test_data:
        :param training_data:
        :param epochs:
        :param mini_batch_size:
        :param eta:
        :return:
        """
        n_test = 0
        training_data = list(training_data)
        # 训练数据的总个数
        n = len(training_data)

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        # 开始训练 循环每一个epochs
        for j in range(epochs):
            # 洗牌 打乱训练数据
            random.shuffle(training_data)
            # mini_batch
            mini_batches = [
                training_data[k: k + mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            # 训练 mini_batch
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                print("Epoch {} : {} / {}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """
        更新 mini_batch
        :param mini_batch:
        :param eta:
        :return:
        """
        # 保存每一层的偏导数
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # 训练每一个 mini_batch
        for x, y in mini_batch:
            delta_nable_b, delta_nable_w = self.back_prop(x, y)

            # 保存一次
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nable_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nable_b)]
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def back_prop(self, x, y):
        """
        反向传播
        :param x:
        :param y:
        :return:
        """
        # 保存每一层的偏导数
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # 前向过程
        activation = x
        # 保存每一层的激励值 a = sigmoid(z)
        activations = [x]
        # 保存每一层的 z = wx + b
        zs = []
        # 前向传播
        for b, w in zip(self.biases, self.weights):
            # 计算每一层的 z
            z = np.dot(w, activation) + b
            # 保存每一层的 z
            zs.append(z)
            # 计算每一层的激励值 a
            activation = sigmoid(z)
            # 保存每一层的 a
            activations.append(activation)

        # 反向过程
        # 计算最后一层的误差
        delta = cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        # 最后一层的权重和偏置的导数
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # 倒数第二层直至第一层的权重和偏置的导数
        for layer in range(2, self.num_layers):
            z = zs[-layer]
            sp = sigmoid_prime(z)
            # 当前层的误差
            delta = np.dot(self.weights[-layer + 1].transpose(), delta) * sp
            # 当前层偏置和权重的导数
            nabla_b[-layer] = delta
            nabla_w[-layer] = np.dot(delta, activations[-layer - 1].transpose())

        return nabla_b, nabla_w

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feed_forward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)


# 定义神经网络结构
def cost_derivative(out_activation, y):
    return out_activation - y


def sigmoid(z):
    """
    激励函数
    :param z:
    :return:
    """
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """
    激励函数的导数
    :param z:
    :return:
    """
    return sigmoid(z) * (1 - sigmoid(z))
