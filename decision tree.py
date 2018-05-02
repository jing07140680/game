from math import log
import csv
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

def createDataSet(TicTacToe):
    csv_file = csv.reader(open(TicTacToe, 'r'))
    dataSet = []
    for turn in csv_file:
        dataSet.append(turn)
    labels = dataSet[0]
    del dataSet[0]
    return dataSet, labels


def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1

    for i in range(numFeatures):

        featList = [example[i] for example in dataSet]

        uniqueVals = set(featList)
        newEntropy = 0.0

        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i;
    return bestFeature

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList):  # 第一个停止条件
        return classList[0]  # stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1:  # stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(
        dataSet)  # the return is an Int,which is a place of the row of the column`s feature，The best feature
    bestFeatLabel = labels[bestFeat]  # 特征中的一个
    myTree = {bestFeatLabel: {}}  # 创建一个树的字典
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]  # featValues是得到当前这一列特征的所有表达
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]  # copy all of labels, so trees don't mess up existing labels，在python中，参数是列表类型时，是按照引用方式传递，会改变他的值
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
        # value的值是上字典key bestFeatLable对应的值，但同时value也是一个字典，value是该字典的key，等号右边的部分是key下的值，该值同时可能又是一个字典
    return myTree

def classify(inputTree,featLabels,testVec):#inputtree是已经生成了的树，testvec是测试向量，featlable是测试向量对应位置的标签
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]#判断测试向量在该分类下的值是否是决策树的叶节点，如果不是则继续递归判断，如果是，返回分类结果
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel