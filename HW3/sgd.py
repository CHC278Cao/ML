# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def cost_function(x, y, weight, bias):
    error = 0
    for i in range(len(x)):
        error += (y[i] - (weight * x[i] + bias)) ** 2
    return error / len(x)

def update_weights(x, y, weight, bias, learning_rate):
    weight_deriv = 0
    bias_deriv = 0
    num = len(x)

    for i in range(num):
        weight_deriv += -2 * x[i] * (y[i] - (weight * x[i] + bias))
        bias_deriv += -2 * (y[i] - (weight * x[i] + bias))

    weight -= (weight_deriv) / num * learning_rate
    bias -= (bias_deriv) / num * learning_rate

    return (weight, bias)

def train(x, y, weight, bias, learning_rate, iter):
    cost_history = []

    for i in range(iter):
        weight, bias = update_weights(x, y, weight, bias, learning_rate)
        cost = cost_function(x, y, weight, bias)
        cost_history.append(cost)
        print("Iter: {}, weight: {}, bias: {}, cost: {}".format(i, weight, bias, cost))


    return weight, bias, cost_history

def figPlot(data):
    xLength = list(range(1, len(data)+1))

    plt.figure(figsize = (10, 10))
    plt.figure(figsize = (10, 10))
    plt.plot(xLength, data, c = 'red', linestyle = '-', linewidth = 1.0)
    plt.show()


def main():
    x = [1, 2, 3, 4, 5]
    y = [3, 8, 9, 12, 15]
    weight = 2
    bias = 0
    learning_rate = 0.001
    iter = 20
    wight, bias, cost = train(x, y, weight, bias, learning_rate, iter)
    print("weight: {}".format(weight))
    print("bias: {}".format(bias))

    figPlot(cost)

if __name__ == "__main__":
    main()