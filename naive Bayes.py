import csv
import numpy
from numpy import *
#read TicTacToe.csv

def readcsv (TicTacToe) :
    csv_file = csv.reader(open(TicTacToe,'r'))
    #rewirte data,and append them into one list
    list = []#original list
    Category = []# original list's category
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
                Category.append(1)
                del turn[index]
            elif element == 'negative':
                numNeg = numNeg + 1
                Category.append(0)
                del turn[index]
            else:
                break
        list.append(turn)
    del list[len(list)-1] #remove the last line
    del list[0] #remove the first line (Gamen)
    xlist, olist = [],[]
    for turn in list:
        turn = [0 if x == 1 else x for x in turn]
        turn = [1 if x == 2 else x for x in turn]
        xlist.append(turn)
    for turn in list:
        turn = [0 if x == 2 else x for x in turn]
        olist.append(turn)
    #print(list,Category,xlist,olist)
    return (list,Category,xlist,olist,numPos,numNeg)

def Train(trainset, traincategory, xtrain, otrain ,numPos,numNeg):
    numtrainset = len(trainset)
    numfeature = len(trainset[0])
    ppos = numPos/float(numtrainset)#ppos = probability of positive in Trainlist
    pneg = numNeg/float(numtrainset) #pneg = probability of negative in Trainlist
    pxpnum = ones(numfeature); pxnnum = ones(numfeature) # pxpnum = probability of positive depends on x position
    popnum = ones(numfeature); ponnum = ones(numfeature)# popnum = probability of positive depends on o position
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
    result = [] #postive is 1, negative is 0
    for i in range(len(xvalidatedata)):
        pp = sum( multiply(pxpVect,xvalidatedata[i]))+sum( multiply(popVect,ovalidatedata[i]))+log(ppos) #the validatedata's the probability of positive
        pn = sum( multiply(pxnVect,xvalidatedata[i]))+sum( multiply(ponVect,ovalidatedata[i]))+log(pneg) #the validatedata's the probability of negative
        #print(pp,pn)
        if pp > pn:
            result.append(1)
        else:
            result.append(0)
    return (result)

def validating():
    list, Category, xlist, olist, numPos,numNeg = readcsv('TicTacToetrain.csv')
    pxpV, pxnV, popV, ponV, ppos, pneg = Train(list, Category, xlist, olist, numPos,numNeg )
    vlist, vCategory, xvlist,ovlist , vnumPos, vnumNeg = readcsv('TicTacToevalidate.csv')
    vresult = classifyTicTacToe(pxpV, pxnV, popV, ponV,xvlist,ovlist, ppos, pneg)
    del vCategory[0]
    TwoVect =ones(len(vCategory))*2 # array
    vresult = numpy.array(vresult) # list transfers to array
    vCategory = numpy.array(vCategory)
    a = vresult - TwoVect
    a = a.astype(int)
    b = vCategory - TwoVect
    b = b.astype(int)
    vvresult =~ a
    vvCategory = ~ b
    PP = sum(vCategory & vresult)
    NN = sum(vvCategory & vvresult)
    PN = sum( vCategory & vvresult)
    NP = sum( vvCategory & vresult)
    print (PP, PN, NP, NN)

validating()