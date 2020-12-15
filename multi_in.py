import numpy as np
from keras.layers import Conv1D  # Convolution Operation
from keras.layers import MaxPooling1D # Pooling
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras.layers import Input, concatenate
from keras.optimizers import Adam, SGD
from keras import Model
from sklearn.preprocessing import OneHotEncoder
import time

def one_in(filter_num = 16, input_shape = 180):
    input1 = Input(shape=(input_shape,1))

    x = Conv1D(filter_num, kernel_size = 2, activation = "relu")(input1)
    x = Conv1D(filter_num, kernel_size = 2, activation = "relu")(x)
    #x = MaxPooling1D(pool_size=2)(x)

    x1 = Conv1D(filter_num*2, kernel_size = 2, activation = "relu")(x)
    x1 = Conv1D(filter_num*2, kernel_size = 2, activation = "relu")(x1)
    #x1 = MaxPooling1D(pool_size=2)(x1)

    x2 = Conv1D(filter_num*4, kernel_size = 2, activation = "relu")(x1)
    x2 = Conv1D(filter_num*4, kernel_size = 2, activation = "relu")(x2)
    #x2 = MaxPooling1D(pool_size=2)(x2)

    x3 = Conv1D(filter_num*6, kernel_size = 2, activation = "relu")(x2)
    x3 = Conv1D(filter_num*6, kernel_size = 2, activation = "relu")(x3)

    x4 = Conv1D(filter_num*8, kernel_size = 2, activation = "relu")(x3)
    flat = Flatten()(x4)
    D1 = Dense(units= 32, activation= "relu")(flat)
    D2 = Dense(units= 5, activation="softmax")(D1)
    return Model(inputs=input1, outputs=D2)

def two_in(filter_num = 32, input_shape1=180, input_shape2=180):
    input1 = Input(shape=(input_shape1,1))

    x = Conv1D(filter_num, kernel_size = 2, activation = "relu")(input1)
    x = Conv1D(filter_num, kernel_size = 2, activation = "relu")(x)
    #x = MaxPooling1D(pool_size=2)(x)

    x1 = Conv1D(filter_num*2, kernel_size = 2, activation = "relu")(x)
    x1 = Conv1D(filter_num*2, kernel_size = 2, activation = "relu")(x1)
    #x1 = MaxPooling1D(pool_size=2)(x1)

    x2 = Conv1D(filter_num*4, kernel_size = 2, activation = "relu")(x1)
    x2 = Conv1D(filter_num*4, kernel_size = 2, activation = "relu")(x2)
    #x2 = MaxPooling1D(pool_size=2)(x2)

    x3 = Conv1D(filter_num*8, kernel_size = 2, activation = "relu")(x2)
    x3 = Conv1D(filter_num*8, kernel_size = 2, activation = "relu")(x3)

    x4 = Conv1D(filter_num*16, kernel_size = 2, activation = "relu")(x3)

    input2 = Input(shape=(input_shape2,1))

    y = Conv1D(filter_num, kernel_size = 3, activation = "relu")(input2)
    y = Conv1D(filter_num, kernel_size = 3, activation = "relu")(y)
    y = Conv1D(int(filter_num/2), kernel_size = 1, activation = "relu")(y)
    y = MaxPooling1D(pool_size=2)(y)

    y1 = Conv1D(filter_num*2, kernel_size = 3, activation = "relu")(y)
    y1 = Conv1D(filter_num*2, kernel_size = 3, activation = "relu")(y1)
    y1 = Conv1D(filter_num, kernel_size = 1, activation = "relu")(y1)
    y1 = MaxPooling1D(pool_size=2)(y1)

    y2 = Conv1D(filter_num*4, kernel_size = 3, activation = "relu")(y1)
    y2 = Conv1D(filter_num*4, kernel_size = 3, activation = "relu")(y2)
    y2 = Conv1D(filter_num*2, kernel_size = 1, activation = "relu")(y2)
    y2 = MaxPooling1D(pool_size=2)(y2)

    y3 = Conv1D(filter_num*8, kernel_size = 3, activation = "relu")(y2)
    y3 = Conv1D(filter_num*8, kernel_size = 3, activation = "relu")(y3)
    y3 = Conv1D(filter_num*4, kernel_size = 1, activation = "relu")(y3)
    
    y4 = Conv1D(filter_num*16, kernel_size = 3, activation = "relu")(y3)

    merge = concatenate([x4, y4], axis=1)
    flat = Flatten()(merge)
    D1 = Dense(units= 32, activation= "relu")(flat)
    D2 = Dense(units= 5, activation="softmax")(D1)

    return Model(inputs=[input1, input2], outputs=D2)

from scipy.signal import welch
def frequency(input, sample_rate):
    f, power = welch(x=input, fs=sample_rate)
    return f, power


#PPG_ = np.load('PPG-5-1.npy')
#label = np.load('label-5-1.npy')
PPG_ = np.load('PPG-5-150s-mid.npy')
label = np.load('label-5-150s-mid.npy')



#rand = np.arange(len(PPG_))
#np.random.shuffle(rand)
#PPG_ = PPG_[rand]
#label = label[rand]

from scipy.signal import find_peaks
from scipy import interpolate

interval = []
f_domain = []
sx = np.arange(0, 125*150)
inter = np.zeros((PPG_.shape[0], 180*4))
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



#model = two_in(input_shape1=len(inter[0]), input_shape2=len(f_domain[0]))
model = one_in(input_shape=len(f_domain[0]))
model.compile(optimizer = Adam(lr = 0.0005), loss = 'categorical_crossentropy', metrics=['categorical_accuracy'])
model.summary()
from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=2)



#start = time.time()
#history = model.fit([inter, f_domain], label, epochs=50, validation_split= 0.3, callbacks=[early_stopping], batch_size=16)
history = model.fit(f_domain, label, epochs=50, validation_split= 0.3, callbacks=[early_stopping], batch_size=16)
#end = time.time()
#print(end - start)
model.save("./models/f_domain-150-linear.h5")