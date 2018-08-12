import matplotlib.pyplot as plt
import numpy as np

def linePlot(costs,alpha,batchSize,thresh,stopType):
    name='learning rate:{}/'.format(alpha)
    if batchSize==1:strDescType='stochastic/'
    else:strDescType='batch size:{}/'.format(batchSize)
    name+=strDescType+'decent-stop:'
    if stopType==0:strStop='{} iterations'.format(thresh)
    elif stopType==1:strStop='costs change<{}'.format(thresh)
    elif stopType==1:strStop='gradient norm<{}'.format(thresh)
    name+=strStop
    fig,ax=plt.subplots(figsize=(12,4))
    ax.plot(np.arange(len(costs)),costs,'r')
    ax.set_xlabel('Iterations')
    ax.set_ylabel('cost')
    ax.set_title(name.upper())
    plt.show()

def barPlot(x,y):
    return

def scatterPlot(x,y):
    return

def boxPlot():
    return
