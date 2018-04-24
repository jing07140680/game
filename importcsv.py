import csv
from numpy import *
#read TicTacToe.csv
csv_file = csv.reader(open('TicTacToe.csv','r'))
#rewirte data,and append them into one list
trainlist = []
listCategory = []
numPos,numNeg = 0,0
for turn in csv_file:
    for index, element in enumerate(turn):
        if element == 'x':
            turn[index] = 2
        elif element == 'o':
            turn[index] = 1
        elif element == 'b':
            turn[index] = 0
        elif element == 'positive':
            numPos = numPos + 1
            listCategory.append(1)
            del turn[index]
        elif element == 'negative':
            numNeg = numNeg + 1
            listCategory.append(0)
            del turn[index]
        else:
            break
    trainlist.append(turn)
del trainlist[len(trainlist)-1]
del trainlist[0]
xlist, olist = [],[]
for turn in trainlist:
    turn = [0 if x == 1 else x for x in turn]
    turn = [1 if x == 2 else x for x in turn]
    xlist.append(turn)
for turn in trainlist:
    turn = [0 if x == 2 else x for x in turn]
    olist.append(turn)
print(trainlist,listCategory,xlist,olist)

def Train(trainset, traincategory, xtrain, otrain):
    numtrainset = len(trainset)
    numfeature = len(trainset[0])
    ppos = numPos/float(numtrainset)
    pneg = numNeg/float(numtrainset)
    pxpnum = ones(numfeature); pxnnum = ones(numfeature)
    popnum = ones(numfeature); ponnum = ones(numfeature)
    for i in range(numtrainset):
        if traincategory[i] == 1:
            pxpnum += xtrain[i]
            popnum += otrain[i]
        else:
            pxnnum += xtrain[i]
            ponnum += otrain[i]
    pxpVect = log(pxpnum/(numPos+2))
    pxnVect = log(pxnnum/(numNeg+2))
    popVect = log(popnum/(numPos+2))
    ponVect = log(ponnum/(numNeg+2))
    return pxpVect,pxnVect,popVect,ponVect,ppos,pneg

def classifyTicTacToe (pxpVect,pxnVect,popVect,ponVect,xvalidatedata,ovalidatedata,ppos,pneg):
    pp = sum( multiply(pxpVect,xvalidatedata))+sum( multiply(popVect,ovalidatedata))+log(ppos)
    pn = sum( multiply(pxnVect,xvalidatedata))+sum( multiply(ponVect,ovalidatedata))+log(pneg)
    print(pp,pn)
    if pp > pn:
        return 1
    else:
        return 0

def validating():
    pxpV, pxnV, popV, ponV, ppos, pneg = Train(trainlist, listCategory, xlist, olist)
    xvd= [0,1,0,0,0,0,0,0,0]
    ovd = [0,0,0,0,0,0,0,0,0]
    print('validatetest is classified as：', classifyTicTacToe(pxpV, pxnV, popV, ponV,xvd,ovd,ppos, pneg))

validating()