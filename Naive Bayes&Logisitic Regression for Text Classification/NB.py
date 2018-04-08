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

#Read data set, extract each words in list for each file
def readFile(path):
    files = os.listdir(path)
    vocabulary = []
    dic = {}
    for file in files:
        f = open(path+"/"+file,encoding = "ISO-8859-1")
        text = f.read()
        text = re.sub('[^a-zA-Z]', ' ', text)
        words_list = text.strip().split()
        dic[file] = words_list
        vocabulary.extend(words_list)
    return vocabulary, dic

#Read data set, extract each words without stopwords in list for each file
def readFileStopWords(path, stopwords):
    files = os.listdir(path)
    vocabulary = []
    dic = {}
    for file in files:
        f = open(path+"/"+file,encoding = "ISO-8859-1")
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

#Extract all unique words
def extractKeys(trainspamList,trainhamList):
    return list(set(trainspamList)|set(trainhamList))

#Calculate proir probability and conditional probability, store them into dictionary
def train_NB(countSpam, countHam, SpamFreDic, HamFreDic, SpamList, HamList, keyList):
    priorSpam = 1.0 * countSpam / (countSpam + countHam)
    priorHam = 1.0 * countHam / (countSpam + countHam)
    totalSpam = len(SpamList)
    totalHam = len(HamList)
    totalType = len(keyList)
    addOne = 1
    condiproSpam = {}
    condiproHam = {}

    for word in keyList:
        occurrence = 0
        if word in SpamFreDic:
            occurrence = SpamFreDic[word]
        condprob1 = 1.0 * (occurrence + addOne) / (totalSpam + totalType)
        condiproSpam[word] = condprob1

    for word in keyList:
        occurrence = 0
        if word in HamFreDic:
            occurrence = HamFreDic[word]
        condprob2 = 1.0 * (occurrence + addOne) / (totalHam + totalType)
        condiproHam[word] = condprob2

    return priorSpam, priorHam, condiproSpam, condiproHam

#Classify the type for test data set and calculate the accurancy
def apply_NB(priorSpam, priorHam, condiproSpam, condiproHam, spamDic, hamDic, keyList):
    setDic = [spamDic, hamDic]
    total = len(spamDic) + len(hamDic)
    correct = 0
    for i in range(len(setDic)):
        for fileName in setDic[i]:
            score1 = log(priorSpam)
            score2 = log(priorHam)
            for word in setDic[i][fileName]:
                if word in keyList:
                    score1 += log(condiproSpam[word])
                    score2 += log(condiproHam[word])
            if score1 >= score2 and i == 0:
                correct +=1
            elif score1 <= score2 and i == 1:
                correct +=1
    return 1.0 * correct/total

if __name__ == "__main__":
    trainSpamPath = 'train/spam'
    trainHamPath = 'train/ham'
    testSpamPath = 'test/spam'
    testHamPath = 'test/ham'
    stopWordsPath = r'./stopwords.txt'
    stopwords = getStopWords(stopWordsPath)
    willRemoveStopWords = sys.argv[1]

    if willRemoveStopWords == 'yes':
        trainspamList, trainspamDic = readFileStopWords(trainSpamPath, stopwords)
        trainhamList, trainhamDic = readFileStopWords(trainHamPath, stopwords)
    else:
        trainspamList, trainspamDic = readFile(trainSpamPath)
        trainhamList, trainhamDic = readFile(trainHamPath)

    testSpamList, testspamDic = readFile(testSpamPath)
    testHamList, testhamDic = readFile(testHamPath)

    numSpam = len(trainspamDic)
    numHam = len(trainhamDic)

    spamFreq = Counter(trainspamList)
    hamFreq = Counter(trainhamList)
    keyList = extractKeys(spamFreq, hamFreq)

    priorSpam, priorHam, condprob1, condprob2 = train_NB(numSpam, numHam,spamFreq, hamFreq, trainspamList, trainhamList,keyList)
    auc=apply_NB(priorSpam, priorHam, condprob1, condprob2, testspamDic, testhamDic, keyList)
    print (auc)
