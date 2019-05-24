# -*- coding:utf-8 -*-

import os
import sys
import numpy as np
import learnhmm as lh

def calAccuLoghood(combo, alpha, beta):
    predictComb = []
    correctCnt = 0
    length = len(combo)

    for t in range(length):
        alpha_t = alpha[:, t]
        beta_t = beta[:, t]
        y_t = np.argmax(np.multiply(alpha_t, beta_t))
        if combo[t][1] == y_t:
            correctCnt += 1
        predictComb.append((combo[t][0], y_t))

    accuracy = (correctCnt, length)
    ll = np.log(np.sum(alpha[:, -1]))

    return (predictComb, accuracy, ll)

def calAlphas(combo, wordLength, tagLength, hmmprior, hmmtrans, hmmemit):
    length = len(combo)
    alpha = np.zeros((tagLength, length))

    # calculate the alpha1_j
    x0 = combo[0][0]
    for k in range(tagLength):
        alpha[k, 0] = hmmprior[k] * hmmemit[k, x0]


    for t in range(1, length):
        xt = combo[t][0]
        for k in range(tagLength):
            alpha[k, t] = hmmemit[k, xt] * np.sum([alpha[j, t-1] * hmmtrans[j, k] for j in range(tagLength)])

    return alpha

def calBetas(combo, wordLength, tagLength, hmmprior, hmmtrans, hmmemit):
    length = len(combo)
    beta = np.zeros((tagLength, length))

    # betaT_j = 1
    for k in range(tagLength):
        beta[k, length-1] = 1.0

    for t in range(length-2, -1, -1):
        xt = combo[t+1][0]
        for k in range(tagLength):
            beta[k, t] = sum(hmmemit[j, xt] * beta[j, t+1] * hmmtrans[k, j] for j in range(tagLength))

    return beta

def contPredict(line_prediction, index_to_word, index_to_tag):

    with open(index_to_word, 'r') as f:
        cont_words = f.readlines()
    with open(index_to_tag) as f:
        cont_tag = f.readlines()
    dict_words = {}
    for ix, word in enumerate(cont_words):
        dict_words[ix] = word.strip()
    dict_tag = {}
    for ix, tag in enumerate(cont_tag):
        dict_tag[ix] = tag.strip()

    contentPredict = []
    for line in line_prediction:
        temp = []
        for predict in line:
            temp_string = "" + dict_words[predict[0]] + "_" + dict_tag[predict[1]]
            temp.append(temp_string)
        contentPredict.append(temp)

    return contentPredict

def dataOps(testFile, index_to_word, index_to_tag, hmmprior, hmmtrans, hmmemit, predictFile, metricFile):
    (testComb, wordLength, tagLength) = lh.getComb(testFile, index_to_word, index_to_tag)
    hmmprior = np.loadtxt(hmmprior)
    hmmtrans = np.loadtxt(hmmtrans)
    hmmemit = np.loadtxt(hmmemit)

    line_prediction = []
    line_accuracy = []
    line_ll = []
    for combo in testComb:
        alpha = calAlphas(combo, wordLength, tagLength, hmmprior, hmmtrans, hmmemit)
        beta = calBetas(combo, wordLength, tagLength, hmmprior, hmmtrans, hmmemit)
        (prediction, accuracy, ll) = calAccuLoghood(combo, alpha, beta)
        line_prediction.append(prediction)
        line_accuracy.append(accuracy)
        line_ll.append(ll)

    contentPredict = contPredict(line_prediction, index_to_word, index_to_tag)
    totalCorrectCnt = 0
    totalLength = 0
    for line in line_accuracy:
        totalCorrectCnt += line[0]
        totalLength += line[1]
    averageAccuy = float(totalCorrectCnt / totalLength)
    averageLl = float(sum(line_ll) / len(line_ll))

    with open(predictFile, 'w') as f:
        for line in contentPredict:
            line = ' '.join(line)
            f.write(line + '\n')

    with open(metricFile, 'w') as f:
        f.write("Average Log-LikeLihood: " + str(averageLl) + '\n')
        f.write("Accuracy: " + str(averageAccuy))


def main():
    test_input = sys.argv[1]
    index_to_word = sys.argv[2]
    index_to_tag = sys.argv[3]
    hmmprior = sys.argv[4]
    hmmtrans = sys.argv[5]
    hmmemit = sys.argv[6]
    predictFile = sys.argv[7]
    metricFile = sys.argv[8]

    dataOps(test_input, index_to_word, index_to_tag, hmmprior, hmmtrans, hmmemit, predictFile, metricFile)

if __name__ == "__main__":
    main()