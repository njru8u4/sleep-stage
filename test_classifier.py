import keras
from keras import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam, SGD
import numpy as np
import tensorflow as tf

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
import xgboost as xgb
import time

import read_data
import analysis

from scipy.signal import find_peaks
from scipy import interpolate
from scipy.signal import welch

def to_MLP(feat_train, label):
    mlp = MLPClassifier(hidden_layer_sizes=32, activation='relu', validation_fraction=0.3, solver='adam', early_stopping=False, verbose=True, n_iter_no_change=5, learning_rate='adaptive', learning_rate_init=0.001, max_iter=200)
    x_train, x_test, y_train, y_test = train_test_split(feat_train, label, test_size = 0.2, shuffle = False)
    mlp.fit(x_train, y_train)
    print("MLP train: {}".format(mlp.score(x_train, y_train)))
    print("MLP test: {}".format(mlp.score(x_test, y_test)))


def to_keras(feat_train, label):
    x_train, x_test, y_train, y_test = train_test_split(feat_train, label, test_size = 0.2, shuffle = True)

    input1 = Input(shape=(len(feat_train[0]), ))
    D = Dense(units=32, activation='relu')(input1)
    D1 = Dense(units=4, activation='softmax')(D)
    model = Model(inputs=input1, outputs=D1)
    model.compile(optimizer = Adam(lr = 0.0005), loss = 'categorical_crossentropy', metrics=['categorical_accuracy'])
    model.summary()
    from keras.callbacks import EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    model.fit(x_train, y_train, epochs=100, validation_split= 0.3, callbacks=[early_stopping], batch_size=32)
    model.save("./models/f_domain-150-keras-upsampled.h5")
    print("fit done")
    pred = model.predict(x_test)
    print(analysis.confusion_matrix(y_test, pred, 4))
    print("keras: {}".format(model.evaluate(x_test, y_test)))

def to_LDA(feat_train, label_class):
    x_train, x_test, y_train, y_test = train_test_split(feat_train, label_class, test_size = 0.2, shuffle = False)
    lda = LinearDiscriminantAnalysis()
    lda.fit(x_train, y_train)
    score = lda.score(x_test, y_test)
    print ("lda test: {}".format(score))
    #return

def to_svm(feat_train, label_class):
    x_train, x_test, y_train, y_test = train_test_split(feat_train, label_class, test_size = 0.2, shuffle = False)
    svm = SVC(kernel='rbf', decision_function_shape='ovo', probability=True)
    svm.fit(x_train, y_train)
    print ("fit done")
    score = svm.score(x_test, y_test)
    print ("svm: {}".format(score))

def to_xgb(feat_train, label_class):
    x_train, x_test, y_train, y_test = train_test_split(feat_train, label_class, test_size = 0.2, shuffle = False)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.3, random_state = 7)
    eval = [(x_val, y_val)]
    xb = xgb.XGBClassifier(learning_rate=0.01, max_depth=4)
    xb.fit(x_train, y_train, eval_metric="mlogloss", eval_set=eval, early_stopping_rounds=5)
    print ("fit done")
    
    y_pred = xb.predict(x_train)
    prediction = [round(value) for value in y_pred]
    score = accuracy_score(y_train, prediction)
    print ("xgb train: {}".format(score))
    #score = xb.score(x_val, y_val)
    y_pred = xb.predict(x_val)
    prediction = [round(value) for value in y_pred]
    score = accuracy_score(y_val, prediction)
    print ("xgb val: {}".format(score))
    #score = xb.score(x_test, y_test)
    y_pred = xb.predict(x_test)
    prediction = [round(value) for value in y_pred]
    score = accuracy_score(y_test, prediction)
    print ("xgb test: {}".format(score))
    #print(xb.get_xgb_params())
    return
def frequency(input, sample_rate):
    f, power = welch(x=input, fs=sample_rate)
    return f, power

def read2(do_f):
    inter = np.load("./datas/inter-4-150s-upsampled.npy")
    label = np.load("./datas/label-4-150s-upsampled.npy")
    f_domain = []
    interval = []
    #new_label = []
    if do_f == 1:
        sx = np.arange(0, 125*150)
        for i in range(inter.shape[0]):
            #n_zero = 0
            x = [0]
            temp = []
            for j in range(100*5):
                if inter[i,j] == 0:
                    #n_zero = j
                    break
                temp.append(inter[i,j])
                x.append(x[j] + inter[i,j])
            #print (len(temp))
            #print (len(x))
            func1 = interpolate.UnivariateSpline(x[1:], temp, s=0)
            sy = func1(sx)
            _, p = frequency(sy, 125)
            f_domain.append(p)
            #new_label.append(label[i, :])
            interval.append(sy)

    
    inter = inter.reshape(len(inter), len(inter[0]), 1).astype(int)
    if do_f == 1:
        interval = np.array(interval)
        interval = interval.reshape(len(interval), len(interval[0]), 1)
        f_domain = np.array(f_domain)
        f_domain = f_domain.reshape(len(f_domain), len(f_domain[0]), 1)
    else:
        f_domain = np.zeros((len(inter), 100))
    #new_label = np.array(new_label).astype(int)
    #print(new_label.shape)
    return inter, f_domain, label

def read1():
    PPG_ = np.load('PPG-4-150s.npy')
    label = np.load('label-4-150s.npy')
    #label_class = np.load('label-5-30s-class.npy')
    interval = []
    f_domain = []
    sx = np.arange(0, 125*150)
    inter = np.zeros((PPG_.shape[0], 100*5))
    for i in range(PPG_.shape[0]):
        peaks, _ = find_peaks(PPG_[i,:], distance=40, height=1, width=20)
        distance = peaks[1:-1] - peaks[0:-2]
        inter[i, :len(distance)] = distance
        func1 = interpolate.UnivariateSpline(peaks[1:-1], distance, s=0)
        sy = func1(sx)
        _, p = frequency(sy, 125)
        f_domain.append(p)
        interval.append(sy)


    interval = np.array(interval)
    interval = interval.reshape(len(interval), len(interval[0]), 1)
    inter = inter.reshape(len(inter), len(inter[0]), 1)
    f_domain = np.array(f_domain)
    f_domain = f_domain.reshape(len(f_domain), len(f_domain[0]), 1)
    #label_class = label_class.reshape(len(label_class))
    PPG_ = PPG_.reshape(len(PPG_), len(PPG_[0]), 1)
    return inter, f_domain, PPG_, label

def get_feature(train, model_name):
    model = keras.models.load_model(model_name)
    model_feat = Model(inputs=model.get_layer('input_1').input, outputs=model.get_layer('dense_out').output)
    features = model_feat.predict(train)
    return features

def tests(feat_train, label):
    label_class = []
    for i in range(len(label)):
        label_class.append(np.argmax(label[i]))
    label_class = np.array(label_class)

    now = time.time()
    to_LDA(feat_train, label_class)
    LDA_time = time.time() - now
    print ("LDA time: {}".format(LDA_time))

    now = time.time()
    to_xgb(feat_train, label_class)
    xgb_time = time.time() - now
    print ("xgb time: {}".format(xgb_time))

    now = time.time()
    to_MLP(feat_train, label_class)
    MLP_time = time.time() - now
    print ("MLP time: {}".format(MLP_time))

    now = time.time()
    to_keras(feat_train, label)
    keras_time = time.time() - now
    print ("keras time: {}".format(keras_time))

    now = time.time()
    to_svm(feat_train, label_class)
    SVM_time = time.time() - now
    print ("SVM time: {}".format(SVM_time))


if __name__ == "__main__":
    inter = read_data.read_inter("./datas/inter-4-150s-upsampled.npy", 5)
    inter_feat = get_feature(inter, "./models/inter-4-150-64-up-bn")
    #f_domain = read_data.read_frequency("./datas/inter-4-150s-upsampled.npy", 5)
    f_domain = np.load("./datas/f_domain-4-150s-upsampled.npy")
    f_feat = get_feature(f_domain, "./models/f_domain-4-150-64-up-multiout")
    label = read_data.read_label("datas/label-4-150s-upsampled.npy")
    
    feature = np.concatenate((inter_feat, f_feat), axis=1)
    tests(feature, label)

