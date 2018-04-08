#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 14:00:18 2018

@author: shengyidan
"""
from math import exp
from numpy import *
import os
import re
from math import log
from math import e
import sys
from collections import Counter

#Extract stopwords list
def getStopWords(path):
    stopwords = []
    f = open(path, 'r')
    for line in f.readlines():
        stopwords.append(line.strip())
    return stopwords

#Read data set, extract each words without stopwords in list for each file
def readFileStopWords(path, stopwords):
    files = os.listdir(path)
    nums = 0
    vocabulary = []
    dic = {}
    for file in files:
        f = open(path+"/"+file,encoding = "ISO-8859-1")
        nums += 1
        text = f.read()
        text = re.sub('[^a-zA-Z]', ' ', text)
        words_list = text.strip().split()
        words_new = []
        for word in words_list:
            if word not in stopwords:
                words_new.append(word)
        dic[file] = words_new
        vocabulary.extend(words_new)
    return vocabulary, dic

#Read data set, extract each words in list for each file
def readFile(path):
    files = os.listdir(path)
    nums = 0
    vocabulary = []
    dic = {}
    for file in files:
        f = open(path+"/"+file,encoding = "ISO-8859-1")
        nums += 1
        text = f.read()
        text = re.sub('[^a-zA-Z]', ' ', text)
        words_list = text.strip().split()
        dic[file] = words_list
        vocabulary.extend(words_list)
    return vocabulary, dic

#Get all file types(spam - 0, ham - 1) into list
def getLabels(numSpamFile, numHamFile):
    dataClass = []
    for i in range(numSpamFile):
        dataClass.append(0)
    for j in range(numHamFile):
        dataClass.append(1)
    return dataClass

#Extract all unique words
def extractKeys(trainspamList,trainhamList):
    return list(set(trainspamList)|set(trainhamList))

#Compare each feature(word) if it is in the whole training word lists,
#If the feature exits, marks as 1, otherwise 0.
def featureList(wholeWords, dic):
    whole = list(wholeWords)
    result = []
    for fileN in dic:
        row = [0] * (len(whole))
        for word in whole:
            if word in dic[fileN]:
                row[whole.index(word)] = 1
        #x0 is always 1
        row.insert(0,1)
        result.append(row)
    return result

#Merge spam data and ham data into the whole
def mergeData(dataset1, dataset2):
    dic3 = dataset1.copy()
    dic3.update(dataset2)
    return dic3

#Sigmoid function
def sigmoid(z):
    return 1.0/(1+exp(-z))

#Training LR by gradient descent with L2 regularization, return weights after updating
def trainLR(dataMatIn,classLabels,lambdaN):
    dataSet = mat(dataMatIn)
    labelSet = mat(classLabels).transpose()
    m,n = shape(dataSet)
    alpha = 0.1
    numIteration = 50
    weights = zeros((n,1))
    for k in range(numIteration):
        h = sigmoid(dataSet*weights)
        error = (labelSet - h)
        weights = weights + alpha * dataSet.transpose()* error - alpha*lambdaN*weights
    return weights

#Classify the test data and count the accurancy
def classify(weight,data,numSpam,numHam):
    dataMatrix = mat(data)
    wx =dataMatrix * weight
    correct = 0
    total = numSpam + numHam

    for i in range(numSpam):
        if wx[i][0] < 0.0:
            correct += 1
    for j in range(numSpam+1,total):
        if wx[j][0] > 0.0:
            correct += 1

    print(1.0 * correct/total)
    return wx

if __name__ == "__main__":
    trainSpamPath = r'train/spam'
    trainHamPath = r'train/ham'
    testSpamPath = r'test/spam'
    testHamPath = r'test/ham'
    stopWordsPath = r'./stopwords.txt'
    lamb = float(sys.argv[1])
    willRemoveStopWords = sys.argv[2]

    stopwords = getStopWords(stopWordsPath)
    if willRemoveStopWords == 'yes':
        trainspamList, trainspamDic = readFileStopWords(trainSpamPath,stopwords)
        trainhamList, trainhamDic = readFileStopWords(trainHamPath,stopwords)
    else:
        trainspamList, trainspamDic = readFile(trainSpamPath)
        trainhamList, trainhamDic = readFile(trainHamPath)

    testspamList, testspamDic = readFile(testSpamPath)
    testhamList, testhamDic = readFile(testHamPath)

    wholeWords = extractKeys(trainspamList,trainhamList)
    wholeTrain = mergeData(trainspamDic,trainhamDic)
    wholeTest = mergeData(testspamDic,testhamDic)

    numTrainSpam = len(trainspamDic)
    numTrainHam= len(trainhamDic)
    numTestSpam = len(testspamDic)
    numTestHam = len(testhamDic)

    trainLabels = getLabels(numTrainSpam,numTrainHam)

    trainList = featureList(wholeWords, wholeTrain)
    testList = featureList(wholeWords, wholeTest)

    weight = trainLR(trainList, trainLabels, lamb)
    test = classify(weight, testList, numTestSpam, numTestHam)
