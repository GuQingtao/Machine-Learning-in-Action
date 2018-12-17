
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from numpy import *
import matplotlib.pyplot as plt


# In[2]:


## 导入数据


# In[3]:


def loadDataSet(filename):
    dataMat = []; labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat, labelMat    


# In[4]:


def selectJrand(i,m):
    j=i
    while (j==i):
        j = int(random.uniform(0,m))
    return j


# In[5]:


def clipAlpha(aj,H,L):
    if aj>H:
        aj = H
    if L>aj:
        aj = L
    return aj


# ### SMO函数伪代码
# * 创建一个alpha向量并将其初始化为0向量
# * 当迭代次数小于最大迭代次数时（外循环）
#     对数据集中的每个数据向量（内循环）：
#         如果该数据向量可以被优化：
#             随机选择另外一个数据向量
#             同时优化这两个向量
#             如果这两个向量都不能被优化，退出内循环
#     如果所有向量都没有被优化，增加迭代数目，继续下一次循环
#     

# In[6]:


def smoSimple(dataMatIn, classLabels, C, toler, maxIter):
    '''5个输入参数：数据集，类别标签，常数C，容错率，退出前最大的循环次数'''
    dataMatrix = mat(dataMatIn); labelMat = mat(classLabels).transpose()
    b = 0; m,n = shape(dataMatIn)
    alphas = mat(zeros((m,1)))
    itern = 0
    
    while(itern<maxIter):
        alphaPairChanged = 0
        for i in range(m):
            fXi= float(multiply(alphas, labelMat).T*(dataMatrix*dataMatrix[i,:].T))+b
            Ei = fXi - float(labelMat[i])
            if ((labelMat[i]*Ei < -toler) and (alphas[i]<C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i, m)
                fXj = float(multiply(alphas, labelMat).T*(dataMatrix*dataMatrix[j,:].T))+b
                Ej = fXj - float(labelMat[j])
                
                alphaIold = alphas[i].copy();
                alphaJold = alphas[j].copy();
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j]-alphas[i])
                    H = min(C, C+alphas[j]-alphas[i])
                else:
                    L = max(0, alphas[j]+alphas[i]-C)
                    H = min(C, C+alphas[j]+alphas[i])
                if L==H:
                    print("L==H")
                    continue
                
                eta = 2.0*dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T  
                if eta>=0:
                    print('eta>0')
                    continue
                alphas[j] -= labelMat[j]*(Ei-Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                if (abs(alphas[j] - alphaJold)<0.00001):
                    print('J not moving enough')
                    continue
                
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])
                b1 = b - Ei - labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T -                     labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej - labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T -                     labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (alphas[i]>0) and (alphas[i]<C):
                    b=b1
                elif (alphas[j]>0) and (alphas[j]<C):
                    b=b2
                else:
                    b = (b1+b2)/2.0
                    
                alphaPairChanged += 1
                print('itern: %d, i: %d, pairs changed %d' % (itern, i, alphaPairChanged))
        if (alphaPairChanged==0):
            itern+=1
        else:
            itern = 0
        print('iteration number：', itern)
    return b, alphas    
                        
                    
                        
                
                    


# In[ ]:


dataArr, labelArr = loadDataSet(r'data/testSet.txt')
b, alphas = smoSimple(dataArr, labelArr, 0.6, 0.001, 40)

