# -*- coding:utf-8 -*-
import os
import sys
import numpy as np


def getHmm(fileComb, wordLength, tagLength):
    priorMatrix = np.zeros(tagLength)
    transMatrix = np.zeros((tagLength, tagLength))
    emissMatrix = np.zeros((tagLength, wordLength))

    for comb in fileComb:
        # get the first word of sentenss
        priorMatrix[comb[0][1]] += 1

        # get the tag in each line, transition matrix won't change
        tags = [t[1] for t in comb]
        for a, b in zip(tags[1:], tags[:-1]):
            transMatrix[b][a] += 1

        for c in comb:
            emissMatrix[c[1]][c[0]] += 1

    priorMatrix = (priorMatrix + 1) / np.sum(priorMatrix + 1)
    transMatrix = (transMatrix + 1) / np.sum(transMatrix + 1, axis = 1)[:, np.newaxis]
    emissMatrix = (emissMatrix + 1) / np.sum(emissMatrix + 1, axis = 1)[:, np.newaxis]

    return (priorMatrix, transMatrix, emissMatrix)


def getComb(fileIn, inToWd, inToTg):
    with open(fileIn, 'r') as f:
        dict_lines = f.readlines()

    dict_words = {}
    with open(inToWd, 'r') as f:
        cont_words = f.readlines()
    dict_tags = {}
    with open(inToTg, 'r') as f:
        cont_tags = f.readlines()

    dict_lines = [l.strip() for l in dict_lines]
    for ind, words in enumerate(cont_words):
        dict_words[words.strip()] = ind
    for ind, tags in enumerate(cont_tags):
        dict_tags[tags.strip()] = ind

    comb = []
    for i in range(len(dict_lines)):
        line = dict_lines[i]
        content = line.split()
        temp_comb = []
        for ch in content:
            word, tag = ch.split('_')
            word_index = dict_words[word]
            tag_index = dict_tags[tag]
            temp_comb.append((word_index, tag_index))
        comb.append(temp_comb)

    return comb, len(cont_words), len(cont_tags)


def fileOps(fileIn, inToWd, inToTg, hmmPr, hmmEm, hmmTs):
    (fileComb, wordLength, tagLength) = getComb(fileIn, inToWd, inToTg)
    (prior, transition, emission) = getHmm(fileComb, wordLength, tagLength)

    with open(hmmPr, 'w') as f:
        prior = map(str, map(np.format_float_scientific, prior))
        content = '\n'.join(prior)
        f.write(content)

    with open(hmmTs, 'w') as f:
        for st in transition:
            st = map(str, map(np.format_float_scientific, st))
            st = ' '.join(st)
            f.write(st + '\n')

    with open(hmmEm, 'w') as f:
        for st in emission:
            st = map(str, map(np.format_float_scientific, st))
            st = ' '.join(st)
            f.write(st + '\n')


def main():
    fileIn = sys.argv[1]
    inToWd = sys.argv[2]
    inToTg = sys.argv[3]
    hmmPr = sys.argv[4]
    hmmEm = sys.argv[5]
    hmmTs = sys.argv[6]

    for file in (fileIn, inToWd, inToTg):
        if os.stat(file).st_size == 0:
            raise ValueError("{} does\'t exist ... ".format(file))

    fileOps(fileIn, inToWd, inToTg, hmmPr, hmmEm, hmmTs)

if __name__ == "__main__":
    main()
