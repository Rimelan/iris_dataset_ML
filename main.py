import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import load_data


def plot_histogram(D,L):
    D0=D[:,L == 0]
    D1=D[:,L == 1]
    D2=D[:,L == 2]
    
    features = {
        0: 'Sepal length',
        1: 'Sepal width',
        2: 'Petal length',
        3: 'Petal width'
        }
    
    
    for feID in range(4):
        plt.figure()
        plt.xlabel(features[feID])
        plt.ylabel('Density')
        plt.hist(D0[feID,:],bins = 10, density= True,label = 'setosa')
        plt.hist(D1[feID,:],bins = 10, density= True,label = 'versicolor')
        plt.hist(D2[feID,:],bins = 10, density= True,label = 'virginica')
        plt.savefig('hist_%d.pdf' %feID)
    plt.show() 
    
def plot_scatter(D,L):
    D0=D[:,L == 0]
    D1=D[:,L == 1]
    D2=D[:,L == 2]
    
    features = {
        0: 'Sepal length',
        1: 'Sepal width',
        2: 'Petal length',
        3: 'Petal width'
        }
    for x1 in range(4):
        for x2 in range(4):
            if x1 == x2:
                continue
            plt.figure()
            plt.xlabel(features[x1])
            plt.ylabel(features[x2])
            plt.scatter(D0[x1,:],D0[x2,:],label = 'setosa')
            plt.scatter(D1[x1,:],D1[x2,:],label = 'versicolor')
            plt.scatter(D2[x1,:],D2[x2,:],label = 'virginica')
            plt.savefig('hist_%d_%d.pdf' %(x1,x2))
    plt.show()
def calculate_mean(D):
    m= np.mean(D,axis=1)
    return np.array(m).reshape(4,1)

if __name__ == '__main__':
    data,labels = load_data.load_data("iris.csv")
    plot_histogram(data, labels)
    plot_scatter(data,labels)
    mean = calculate_mean(data)
    var = data.var(1)
    std = data.std(1)
    DCent = data - mean
    covar = (DCent @ DCent.T)/float(data.shape[1])
    
    
    for clss in [0,1,2]:
        print('Class',clss)
        DClass = data[:, labels==clss]
        mean_class= calculate_mean(DClass)
        print('Mean: ', mean_class)
        var_class = DClass.var(1)
        std_class = DClass.std(1)
        print('Variance: ',var_class) 
        print('Standart deviation: ',std_class)
        covar_class = ((DClass-mean_class)@(DClass-mean_class).T)/(float(DClass.shape[1]))
        print('covar: ', covar_class)
    
    
