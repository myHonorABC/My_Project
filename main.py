import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import logistic
import dataPlot


def processData():
    #基于初始数据绘制散点图
    pdData=pd.read_csv('data.txt',header=None,names=['x1','x2','y'])
    positive=pdData[pdData['y']==1]
    negative=pdData[pdData['y']==0]
    fig,ax=plt.subplots(figsize=(10,5))
    ax.scatter(positive['x1'],positive['x2'],s=30,c='b',marker='o',label='admitted')
    ax.scatter(negative['x1'],negative['x2'],s=30,c='r',marker='x',label='not admitted')
    ax.legend()
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    plt.show()

    #data=pdData.as_matrix()
    data=pdData.values        #将dataframe结构数据转化为ndarray结构数据
    theta=np.ones([1,2])      #迭代前的初始值
    batchSize=3               #采样样本的大小
    stopType=0                #停止策略
    thresh=100                #迭代停止阈值
    alpha=0.1                 #学习率
    theta,step,costs,grad,total_time=logistic.descent(data,theta,batchSize,stopType,thresh,alpha)  #开始学习
    print('\ntheta:%s\nstep:%s\ncosts:%s\ngrad:%s\ntotal_time:%s\n'%(theta,step,costs,grad,total_time))
    dataPlot.linePlot(costs,alpha,batchSize,thresh,stopType)      #绘制损失函数曲线

if __name__=='__main__':
    processData()

