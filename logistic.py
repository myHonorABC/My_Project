import numpy as np
import time
 

#sigmoid函数
def sigmoid(z):
    return 1.0/(1+np.exp(-z))

#模型
def model(x,theta):
    return sigmoid(np.dot(x,theta.T))

#损失值
def cost(x,y,theta):
    left=-y*np.log(model(x,theta))
    right=(1-y)*np.log(1-model(x,theta))
    return np.sum(left-right)/len(x)

#计算梯度
def gradient(x,y,theta):
    grad=np.zeros(theta.shape)
    error=(model(x,theta)-y).ravel()          #误差
    for j in range(len(theta.ravel())):       #计算theta的梯度
        term=np.multiply(error,x[:,j])
        grad[0,j]=np.sum(term)/len(x)
        return grad

#停止策略
def stopCriterion(stopType,value,thresh):
    if stopType==0:
        return value>thresh
    elif stopType==1:
        return abs(value[-1]-value[-2])<thresh
    elif stopType==2:
        return np.linalg.norm(value)<thresh

#打乱样本顺序
def shuffleData(data):
    np.random.shuffle(data)
    cols=data.shape[1]
    x=data[:,0:cols-1]
    y=data[:,cols-1:cols]
    return x,y

#使用梯度下降法对样本进行分类
def descent(data,theta,batchSize,stopType,thresh,alpha):
    init_time=time.time()
    i=0
    k=0
    x,y=shuffleData(data)
    grad=np.zeros(theta.shape)
    costs=[cost(x,y,theta)]
    while True:
        grad=gradient(x[k:k+batchSize,:],y[k:k+batchSize,:],theta)
        k+=batchSize
        if k>=data.shape[0]:
            k=0
            x,y=shuffleData(data)
            theta=theta-alpha*grad
            costs.append(cost(x,y,theta))
            i+=1
        if stopType==0:
            value=i
        elif stopType==1:
            value=costs
        elif stopType==2:
            value=grad
        if stopCriterion(stopType,value,thresh):
            break
    return theta,i-1,costs,grad,time.time()-init_time


