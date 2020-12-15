import keras
from keras import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam, SGD
import numpy as np
import tensorflow as tf
from scipy.signal import find_peaks


PPG_ = np.load('PPG-5-30s.npy')
label = np.load('label-5-30s.npy')
label_class = np.load('label-5-30s-class.npy')


interval = []

inter = np.zeros((PPG_.shape[0], 180*1))
for i in range(PPG_.shape[0]):
    peaks, loca = find_peaks(PPG_[i,:], distance=40, height=1, width=20)
    distance = peaks[1:-1] - peaks[0:-2]
    inter[i, :len(distance)] = distance
    
inter = inter.reshape(len(inter), len(inter[0]), 1)

model = keras.models.load_model('5-30-inter.h5')
pred = model.predict(inter)

matrix = np.zeros((5,5), dtype=int)
actual = np.zeros(5, dtype=int)
preds = np.zeros(5, dtype=int)
for i in range(len(pred)):
    actual[np.argmax(label[i])] += 1
    preds[np.argmax(pred[i])] += 1
    matrix[np.argmax(pred[i]),np.argmax(label[i])] += 1

print ("actual {}".format(actual))
print ("pred {}".format(preds))
print (matrix)