# -*- coding:utf-8 -*-

import os
import sys



def createDict(dictFile):
    word_dict = {}
    with open(dictFile, 'r') as f:
        lines = f.readlines()

    for line in lines:
        data = line.strip().split(' ')
        word_dict[data[0]] = data[1].strip()

    return word_dict


def createLabel(data, wordDict, threshold):
    target_dict = {}
    for word in data:
        try:
            new_key = wordDict[word]
        except KeyError:
            continue
        if new_key in target_dict:
            target_dict[new_key] += 1
        else:
            target_dict[new_key] = 1

    if threshold != 0:
        target_dict = {k: 1 for k, v in target_dict.items() if v < threshold}
    else:
        target_dict = {k: 1 for k, v in target_dict.items()}

    return target_dict


def parseFile(fileIn, fileOut, wordDict, threshold):

    with open(fileIn, 'r') as f:
        content = f.readlines()

    parse_line = ""
    for line in content:
        line.strip()
        label, words = line.split('\t')
        parse_line += label
        words = words.split()
        new_dict = createLabel(words, wordDict, threshold)
        for k, v in new_dict.items():
            parse_line += '\t' + str(k) + ':' + str(v)
        parse_line += '\n'

    with open(fileOut, 'w') as f:
        f.write(parse_line)


def fileOps(trainFileIn, validFileIn, testFileIn, dictFile, trainFileOut, validFileOut, testFileOut, threshold):
    word_dict = createDict(dictFile)
    parseFile(trainFileIn, trainFileOut, word_dict, threshold)
    parseFile(validFileIn, validFileOut, word_dict, threshold)
    parseFile(testFileIn, testFileOut, word_dict, threshold)


def main():
    trainFileIn = sys.argv[1]
    validFileIn = sys.argv[2]
    testFileIn = sys.argv[3]
    dictFile = sys.argv[4]
    trainFileOut = sys.argv[5]
    validFileOut = sys.argv[6]
    testFileOut = sys.argv[7]
    mode = int(sys.argv[8])
    for file in (trainFileIn, validFileIn, testFileIn, dictFile):
        if os.stat(file).st_size == 0:
            raise ValueError('{} doesn\'t exist ...')

    if mode == 1:
        fileOps(trainFileIn, validFileIn, testFileIn, dictFile, trainFileOut, validFileOut, testFileOut, 0)
    if mode == 2:
        fileOps(trainFileIn, validFileIn, testFileIn, dictFile, trainFileOut, validFileOut, testFileOut, 4)

if __name__ == '__main__':
    main()



