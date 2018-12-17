import operator
import numpy as np
from os import listdir

def train_data(filename):
    X = []
    Y = []
    with open(filename,'r') as f:
        for line in f.readlines():
            data = line.rstrip('\n').split('\t')
            X.append(list(map(float,(data[0:-1]))))
            Y.append(int(data[-1]))
    return np.array(X),np.array(Y)


def Normalization(X):
    Xmin = np.min(X,axis=0)
    Xmax = np.max(X,axis=0)
    X = (X-Xmin)/(Xmax-Xmin)
    return X
'''辅助函数'''    
def distance_com(data,X):
    m = X.shape[0]
    distance = np.sum((np.tile(data,(m,1))-X)**2,axis=1)**0.5
    return distance

def find_top_k(distance,Y,k):
    sort_index = np.argsort(distance)
    class_count = {}
    for i in range(k):
        index = sort_index[i]
        class_count[Y[index]] = class_count.get(Y[index],0)+1
    return class_count

def find_max_prob(class_count):
    sort_class = sorted(class_count.items(),key=operator.itemgetter(1),reverse=True)
    return sort_class[0][0]    

def classifier(data,X,Y,k=3):
    distance = distance_com(data,X)
    class_count = find_top_k(distance,Y,k)
    pred = find_max_prob(class_count)
    return pred
       

def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect



trainingFileList = listdir('trainingDigits')           #load the training set
m = len(trainingFileList)
trainingMat = np.zeros((m, 1024))
hwLabels = []
for i in range(m):
    fileNameStr = trainingFileList[i]
    fileStr = fileNameStr.split('.')[0]     #take off .txt
    classNumStr = int(fileStr.split('_')[0])
    hwLabels.append(classNumStr)
    trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
testFileList = listdir('testDigits')        #iterate through the test set
errorCount = 0.0
mTest = len(testFileList)
for i in range(mTest):
    fileNameStr = testFileList[i]
    fileStr = fileNameStr.split('.')[0]     #take off .txt
    classNumStr = int(fileStr.split('_')[0])
    vectorUnderTest = img2vector('testDigits/{}'.format(fileNameStr))
    classifierResult = classifier(vectorUnderTest, trainingMat, hwLabels, 3)
    print("the classifier came back with: {}, the real answer is: {}".format(classifierResult, classNumStr))
    if (classifierResult != classNumStr): errorCount += 1.0
print("\nthe total number of errors is: {}".format(errorCount))
print("\nthe total error rate is: {}" .format(errorCount/float(mTest)))









