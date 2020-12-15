import numpy as np
from numba import jit
from sklearn import metrics
from read_data import read_frequency
#def kappa(data):

def confusion_matrix(actual, predict, classes):
    real = np.zeros(actual.shape[0], dtype=int)
    pred = np.zeros(predict.shape[0], dtype=int)
    
    for i in range(predict.shape[0]):
        real[i] = np.argmax(actual[i])
        pred[i] = np.argmax(predict[i])

    param = {'matrix': metrics.confusion_matrix(real, pred),
            'kappa': metrics.cohen_kappa_score(real, pred),
            'accuracy': metrics.accuracy_score(real, pred),
            }
    #print (real)
    precision = []
    recall = []
    for i in range(classes):
        precision.append(param['matrix'][i,i] / sum(param['matrix'][:, i]))
        recall.append(param['matrix'][i,i] / sum(param['matrix'][i, :]))
    param['precision'] = np.array(precision)
    param['recall'] = np.array(recall)

    return param


if __name__ == "__main__":
    f_domain = read_frequency("./datas/inter-4-150s-upsampled.npy", time_stamp=5)

