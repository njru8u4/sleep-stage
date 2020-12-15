import numpy as np
from keras.layers import Conv1D  # Convolution Operation
from keras.layers import MaxPooling1D, AveragePooling1D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras.layers import Input, concatenate
from keras.optimizers import Adam, SGD
from keras import Model
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import time


def tcn(input, filter_num):

    x1 = Conv1D(filter_num, kernel_size = 3, activation = "relu")(input)
    x2 = Conv1D(filter_num, kernel_size = 3, activation = "relu", dilation_rate = 2)(x1)
    x3 = Conv1D(filter_num, kernel_size = 3, activation = "relu", dilation_rate = 4)(x2)
    x4 = Conv1D(filter_num, kernel_size = 3, activation = "relu", dilation_rate = 8)(x3)
    x5 = Conv1D(filter_num, kernel_size = 3, activation = "relu", dilation_rate = 16)(x4)
    #x6 = Dropout(0.3)(x5)
    x6 = Conv1D(filter_num, kernel_size = 3, padding = "same", activation = "relu", dilation_rate = 6)(x5)
    return x5

PPG_ = np.load('PPG-4-150s-mid.npy')
label = np.load('label-4-150s-mid.npy')

from scipy.signal import find_peaks
from scipy import interpolate
from scipy.signal import welch
import matplotlib.pyplot as plt

def frequency(input, sample_rate):
    f, power = welch(x=input, fs=sample_rate)
    return f, power

interval = []
f_domain = []
sx = np.arange(0, 125*150)
inter = np.zeros((PPG_.shape[0], 100*5))
for i in range(PPG_.shape[0]):
    peaks, loca = find_peaks(PPG_[i,:], distance=40, height=1, width=20)
    distance = peaks[1:-1] - peaks[0:-2]
    inter[i, :len(distance)] = distance
    func1 = interpolate.UnivariateSpline(peaks[1:-1], distance, s=0)
    sy = func1(sx)
    f, p = frequency(sy, 125)
    f_domain.append(p)
    interval.append(sy)


interval = np.array(interval)
interval = interval.reshape(len(interval), len(interval[0]), 1)
inter = inter.reshape(len(inter), len(inter[0]), 1)
f_domain = np.array(f_domain)
f_domain = f_domain.reshape(len(f_domain), len(f_domain[0]), 1)
PPG_ = PPG_.reshape(len(PPG_), len(PPG_[0]), 1)

x_train1, x_test1, y_train1, y_test1 = train_test_split(f_domain, label, test_size = 0.2, shuffle = False)
x_train2, x_test2, y_train1, y_test1 = train_test_split(inter, label, test_size = 0.2, shuffle = False)

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=2)

inputs = Input(shape =(len(inter[0]), 1))

flat = Flatten()(tcn(inputs, 16))
drop = Dropout(0.3)(flat)
D1 = Dense(units=32, activation="relu")(drop)
D2 = Dense(units=4, activation="softmax")(D1)
model = Model(inputs= inputs, outputs = D2)
model.compile(optimizer = Adam(lr = 0.0005), loss = 'categorical_crossentropy', metrics=["categorical_accuracy"])
model.summary()

history = model.fit(inter, label, epochs=100, validation_split= 0.3, callbacks=[early_stopping], batch_size=32)
model.save("./models/inter-150-4-tcn-drop.h5")