import matplotlib.pyplot as plt
import numpy as np

from chapter2 import mnist_loader
from chapter2.neural_network import NetWork


def plot_load_data_image():
    training_data, validation_data, test_data = mnist_loader.load_data()
    im = np.array(training_data[0][4])
    im = im.reshape(28, 28)
    plt.imshow(im, cmap='binary')
    plt.show()


def print_load_data_wrapper():
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    training_data_list = list(training_data)
    print(len(training_data_list[0][0]))
    print(training_data_list[0][1])


def test_network():
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = NetWork([784, 30, 10])
    net.sgd(training_data, 30, 10, 0.5, test_data=test_data)


if __name__ == "__main__":
    # plot_load_data_image()

    print_load_data_wrapper()

    test_network()
