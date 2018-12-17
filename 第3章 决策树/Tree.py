# -*- coding: utf-8 -*-
import numpy as np
import operator

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    return dataSet, labels

def caculate_entropy(dataset):
    '''计算信息熵，度量数据集的无序程度'''
    m = len(dataset)
    labels_count = {}
    for data in dataset:
        label = data[-1]
        labels_count[label] = labels_count.get(label,0)+1   #将数据集按照标签进行统计，结果存为字典
    
    entropy = 0.0
    for key in labels_count.keys():                         #计算信息熵
        prob = float(labels_count[key])/m
        entropy -= prob*np.log2(prob) 
    return entropy

def split_dataset(dataset,axis,values):                      #dataset = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    '''划分数据集dataset,axis为划分的依据特征,values为需要返回的特征值'''
    reduce_dataset =[]                                      #split_dataset(dataset,0,1)         
    for data in dataset:                                    #[[1,'yes'],[1,'yes'],[0,'no']]
        if data[axis]==values:                              #划分的依据按照轴axis处取值的不同划分为两类
            reduce_data = data[:axis]                       #等于value的一类，否则为另一类
            reduce_data.extend(data[axis+1:])
            reduce_dataset.append(reduce_data)
    return reduce_dataset

def choose_split_feature(dataset):
    feature_num = len(dataset[0])-1
    base_entropy = caculate_entropy(dataset)
    best_info_gain = 0.0;
    best_feature = -1
    
    for i in range(feature_num):
        feature_list = [data[i] for data in dataset]
        unique_feature = set(feature_list)
        
        new_entropy = 0.0
        for feature in unique_feature:
            sub_dataset = split_dataset(dataset,i,feature)
            prob = len(sub_dataset)/float(len(dataset))
            new_entropy += prob*caculate_entropy(sub_dataset)
        info_gain = base_entropy-new_entropy
        
        if (info_gain>best_info_gain):
            best_info_gain = info_gain
            best_feature = i
    return best_feature

def majorityCnt(classlist):
    class_count = {}
    for vote in classlist:
        class_count[vote] = class_count.get(vote,0)+1
    sort_class_count = sorted(class_count.items(),key=operator.itemgetter(1),reverse=True)
    return sort_class_count[0][0]

def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList): 
        return classList[0]#stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1: #stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    bestFeat = choose_split_feature(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]       #copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(split_dataset(dataSet, bestFeat, value),subLabels)
    return myTree                            
    
def classify(inputTree,featLabels,testVec):
    firstStr = inputTree.keys()[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel

def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()
    
def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
    
dataset,labels = createDataSet()
my_tree = createTree(dataset,labels)
print(my_tree)
        






















