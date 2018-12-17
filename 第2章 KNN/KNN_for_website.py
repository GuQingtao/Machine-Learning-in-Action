import operator
import numpy as np


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
       
    
filename = 'datingTestSet.txt'
X,Y = importdata(filename)
X = Normalization(X)

num = int(len(Y)*0.1)

count = 0.0
for i in range(num):
    pred = classifier(X[i],X[num:,:],Y[num:])
    print('true class is {},pred class is {}'.format(Y[i],pred))
    if pred == Y[i]:
        count +=1

accuracy = count/num
print('accuracy = {}'.format(accuracy))









