import numpy as np


def load_data(fname):
    data = []
    labels = []
    hLabels = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
    }
    with open(fname) as f:
        for line in f:
            curr = line.split(',')[0:-1]
            curr = np.array(curr).reshape(4,1)
            data.append(curr)
            name = line.split(',')[-1].strip()
            labels.append(hLabels[name])
    return data,labels