# -*- coding:utf-8 -*-

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

def avg_neg_likelihood(data, labels, weights, bias):
    logRes = 0
    for i in range(len(data)):
        feature = data[i]
        product = getProduct(feature, weights, bias)
        label = labels[i]
        logRes += -label*product + np.log2(1 + np.exp(product))

    return logRes / len(labels)


def getProduct(feature, weights, bias):
    out = 0
    for key, val in feature.items():
        out += val * weights[int(key)]
    out += out + bias
    return out

def calculateGradient(feature, difference, weightsDim):
    gradient = np.zeros((weightsDim, 1))
    for key, val in feature.items():
        gradient[int(key)] = - val * difference
    return gradient


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def getLabel(x):
    z = sigmoid(x)
    label = 1 if z >= 0.5 else 0
    return label


def fit(traindata, trainlabels, weights, bias, num_epochs, learning_rate):
    for i in range(num_epochs):
        for j in range(len(traindata)):
            feature = traindata[j]
            product = getProduct(feature, weights, bias)
            difference = trainlabels[j] - sigmoid(product)
            weightsGrad = calculateGradient(feature, difference, len(weights))
            biasGrad = - difference
            weights -= learning_rate * weightsGrad
            bias -= learning_rate * biasGrad
    return weights, bias


def predict(data, labels, weights, bias):
    labelPred = []
    for i in range(len(data)):
        feature = data[i]
        product =  getProduct(feature, weights, bias)
        label = getLabel(product)
        labelPred.append(label)

    labelMatch = []
    for i in range(len(labels)):
        if labelPred[i] == labels[i]:
            labelMatch.append(1)
        else:
            labelMatch.append(0)

    print(labelMatch)
    error = sum(labelMatch) / len(labelMatch)
    return labelPred, error


def fetchData(trainFileIn):
    with open(trainFileIn, 'r') as f:
        content = f.readlines()
    labels = []
    data = []
    for line in content:
        temp = {}
        line = line.strip()
        x = line.split('\t')
        labels.append(int(x[0]))
        for word in x[1:]:
            y = word.split(':')
            temp[y[0]] = int(y[1])
        data.append(temp)

    return labels, data

def figPlot(traindata, trainlabels, validdata, validlabels, weights, bias, num_epochs, learning_rate):

    iter = []
    trainLikelihood = []
    validLikelihood = []
    for i in range(1, num_epochs+1):
        trainedweights, trainedbias = fit(traindata, trainlabels, weights, bias, i, learning_rate)
        trainlog = avg_neg_likelihood(traindata, trainlabels, trainedweights, trainedbias)
        validlog = avg_neg_likelihood(validdata, validlabels, trainedweights, trainedbias)
        iter.append(i)
        trainLikelihood.append(trainlog)
        validLikelihood.append(validlog)

    fig = plt.figure(figsize = (12, 12))
    plt.plot(iter, trainLikelihood, c = 'r', label = "train")
    plt.plot(iter, validLikelihood, c = 'y', label = "valid")
    plt.xlabel("Iteration")
    plt.ylabel("Avg_log_neg_likelihood")
    plt.legend(loc = 'upper right')
    plt.show()



def main():
    trainFileIn = sys.argv[1]
    validFileIn = sys.argv[2]
    testFileIn = sys.argv[3]
    dictFile = sys.argv[4]
    trainFileOut = sys.argv[5]
    testFileOut = sys.argv[6]
    matricsOut = sys.argv[7]
    num_epoch = int(sys.argv[8])

    for file in (trainFileIn, validFileIn, testFileIn, dictFile):
        if os.stat(file).st_size == 0:
            raise ValueError('{} does\'t exist ...'.format(file))

    # parse file to get the desired labels and desired data
    trainlabels, traindata = fetchData(trainFileIn)
    validlabels, validdata = fetchData(validFileIn)
    testlabels, testdata = fetchData(testFileIn)

    # initialize the weights and bias, weights size equals the wordDict length
    with open(dictFile, 'r') as f:
        content = f.readlines()
    weights = np.zeros((len(content), 1))
    print(weights.shape)
    bias = 0

    # weights and bias after training data
    trainedweights, trainedbias = fit(traindata, trainlabels, weights, bias, num_epoch, learning_rate = 0.1)
    figPlot(traindata, trainlabels, validdata, validlabels, weights, bias, 200, learning_rate = 0.1)

    # # test
    new_train_labels, train_accuracy = predict(traindata, trainlabels, trainedweights, trainedbias)
    new_valid_labels, valid_accuracy = predict(validdata, validlabels, trainedweights, trainedbias)
    new_test_labels, test_accuracy = predict(testdata, testlabels, trainedweights, trainedbias)
    #
    print("train_accuracy: {}".format(train_accuracy))
    print("new_valid_labels: \n{}".format(new_valid_labels))
    print("valid_accuracy: {}".format(valid_accuracy))
    print("test_accuracy: {}".format(test_accuracy))

    new_train_labels = '\n'.join(map(str, new_train_labels))
    new_test_labels = '\n'.join(map(str, new_test_labels))

    # with open(trainFileOut, 'w') as f:
    #     f.write(new_train_labels)
    #
    # with open(testFileOut, 'w') as f:
    #     f.write(new_test_labels)
    #
    # with open(matricsOut, 'w') as f:
    #     f.write("error(train): {}\n".format(1 - train_accuracy))
    #     f.write("error(test): {}".format(1 - test_accuracy))

if __name__ == '__main__':
    main()