# coding: UTF-8

import numpy as np
import random

def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('Ch05_testSet.txt')
    
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    
    return dataMat, labelMat
    
    
def sigmoid(inX):
    return 1.0 / (1 + np.exp(- inX))


def gradAscent(dataMatIn, classLabels):
    # 转化为矩阵数据类型 
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    #  w = w +or- alpha * gradient 
     # m个数据, 特征数为 n
    m, n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n, 1))
    
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.T * error
        
    return weights
    

def stocGradAscent0(dataMatrix, classLabels, numIter=150): # stochastic 随机
    #  w = w +or- alpha * gradient 
    # m 个数据, n 个特征    
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numIter):    # j 代表迭代次数
        dataIndex = list(range(m))
        for i in range(m):      # i 代表样本下标
            # alpha 每次迭代时需要调整; 
             # alpha是越来越小的, 但j<max(i)时, alpha就不是严格下降的,
            alpha = 4 / (1.0 + j +i) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            # 随机选取样本更新回归系数
            h = sigmoid(np.sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del (dataIndex[randIndex])
        
    return weights
        

# 分析数据: 画出决策边界
def plotBestFit(wei):
    import matplotlib.pyplot as plt
    # getA 矩阵类型转化为数组
    # weights = wei.getA()
    weights = wei
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1]); ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1]); ycord2.append(dataArr[i, 2])
    # 画图 这里画图有的数据类型都是arr
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='r', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()
    
    
def classifyVector(inX, weights):
    prob = sigmoid(np.sum(inX * weights))
    if prob > 0.5: return 1.0
    else: return 0.0
    

def colicTest():
    frTrain = open('Ch05_horseColicTraining.txt')
    frTest = open('Ch05_horseColicTest.txt')
    
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        curLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(curLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(curLine[21]))
    trainWeights  = stocGradAscent0(\
                    np.array(trainingSet), trainingLabels, 500)
    
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        curLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(curLine[i]))
        if int(classifyVector(np.array(lineArr), trainWeights)) != \
            int(curLine[21]):
            errorCount += 1
            
    errorRate = (float(errorCount) / numTestVec)
    print(f'the error rate of this test is: {errorRate}')
    return errorRate


def multiTest():
    numTests = 10; errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print('after {:d} iterations the averge error rate is: {:f}'.format(\
                    numTests, errorSum / float(numTests)))
                    
            
    

    

    
    
    



    


