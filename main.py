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
    



if __name__ == '__main__':
    data,labels = load_data.load_data("iris.csv")
    plot_histogram(data, labels)
    print("hi")