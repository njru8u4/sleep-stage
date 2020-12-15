import numpy as np
from keras.layers import Conv1D  # Convolution Operation
from keras.layers import MaxPooling1D, AveragePooling1D
from keras.layers import Flatten, BatchNormalization
from keras.layers import Dense, Dropout
from keras.layers import Input, concatenate
from keras.optimizers import Adam, SGD
from keras import Model
from sklearn.model_selection import train_test_split
import time

import read_data
import analysis
import temp_model

def cell(filter_num, input, pool):
    #x = input
    x = BatchNormalization()(input)
    #x = Conv1D(filter_num*2, kernel_size = 2, padding = "same", activation = "relu", kernel_initializer='normal')(x)
    
    #x = input
    x1 = Conv1D(filter_num, kernel_size = 1, activation = "relu", kernel_initializer='normal')(x)

    x2 = Conv1D(filter_num*4, kernel_size = 5, padding = "same", activation = "relu", kernel_initializer='normal')(x)
    x2 = Conv1D(filter_num, kernel_size = 1, padding = "same", activation = "relu", kernel_initializer='normal')(x2)

    x3 = Conv1D(filter_num*4, kernel_size = 2, padding = "same", activation = "relu", kernel_initializer='normal')(x)
    x3 = Conv1D(filter_num, kernel_size = 1, padding = "same", activation = "relu", kernel_initializer='normal')(x3)

    x4 = Conv1D(filter_num*4, kernel_size = 3, padding = "same", activation = "relu", kernel_initializer='normal')(x)
    x4 = Conv1D(filter_num, kernel_size = 1, activation = "relu", kernel_initializer='normal')(x4)

    #x5 = Conv1D(filter_num, kernel_size = 1, padding = "same", activation = "relu")(x)
    #x5 = Conv1D(filter_num*2, kernel_size = 2, padding = "same", activation = "relu")(x5)

    merge = concatenate([x1, x2, x3, x4], axis=2)
    #drop = Dropout(0.1)(merge)
    if pool == 1:
        pooling = MaxPooling1D(pool_size = 2)(merge)
    else:
        pooling = AveragePooling1D(pool_size = 2)(merge)

    return pooling

def one_in(input1, pool_type):
    
    #x = Conv1D(64, kernel_size = 3, padding = "same", activation = "relu")(input1)
    #x = BatchNormalization()(input1)
    x = input1
    cell1 = cell(16, x, pool_type)
    cell1 = cell(16, cell1, pool_type)
    cell1 = cell(16, cell1, pool_type)
    #cell1 = cell(8, cell1, pool_type)
    cell1 = Dropout(0.5)(cell1)
    #cell1 = cell(16, cell1, 1)
    #cell1 = cell(16, cell1, 2)
    #cell1 = cell(20, cell1, 2)
    return cell1

def train_model(input_data, label, classes):
    
    input1 = Input(shape=(len(input_data[0]), 1), name = "input_1")
    M = one_in(input1, 2)
    M2 = one_in(input1, 1)
    MM = concatenate([M, M2], axis=2)
    flat = Flatten()(M)
    BN = BatchNormalization()(flat)
    #drop = Dropout(0.5)(flat)
    #drop = BatchNormalization()(drop)
    D = Dense(units=64, activation = "relu", name = "dense_out")(flat)
    D1 = Dense(units = classes, activation = "softmax", name = "output_layer")(D)

    model = Model(inputs= input1, outputs = D1)
    model.compile(optimizer = Adam(lr = 0.0005), loss = 'categorical_crossentropy', metrics=["categorical_accuracy"])
    model.summary()
    from keras.callbacks import EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=2)
    history = model.fit(input_data, label, epochs=100, validation_split= 0.3, callbacks=[early_stopping], batch_size=64)
    return model


def result(train, label, classes, model_name = None):
    print ("start!!")

    x_train1, x_test1, y_train1, y_test1 = train_test_split(train, label, test_size = 0.2, shuffle = True)
    model = train_model(x_train1, y_train1, classes)
    #model = temp_model.frequency_train_model(x_train1, y_train1, classes)
    print (model.evaluate(x_test1, y_test1))

    if model_name != None:
        model.save(model_name)
    pred = model.predict(x_test1)

    results = analysis.confusion_matrix(y_test1, pred, classes)
    return results

def result_multi(train, label, classes, model_name = None):
    print ("start!!")

    x_train1, x_test1, y_train1, y_test1 = train_test_split(train, label, test_size = 0.2, shuffle = True)
    model = temp_model.multi_output(x_train1, y_train1, classes)
    new_model = Model(inputs=model.get_layer("input_1").input, outputs=model.get_layer("output_layer").output)
    if model_name != None:
        model.save(model_name)
    pred = new_model.predict(x_test1)

    results = analysis.confusion_matrix(y_test1, pred, classes)
    return results

if __name__ == "__main__":
    inter = read_data.read_inter("./datas/inter-4-150s-upsampled.npy", time_stamp=5)
    #f_domain = read_data.read_frequency("./datas/inter-4-150s-upsampled.npy", time_stamp=5)
    #np.save("./datas/f_domain-4-150s-upsampled.npy", f_domain)
    #f_domain = np.load("./datas/f_domain-4-150s-upsampled.npy")
    label = read_data.read_label("./datas/label-4-150s-upsampled.npy")
    
    inter_result = result(inter, label, classes=4)
    #inter_result = result(inter, label, classes=4, model_name="./models/inter-4-150-64-up-bn")
    #f_result = result(f_domain, label, classes=4)
    #f_result = result(f_domain, label, classes=4, model_name="./models/f_domain-4-150-up")
    
    #f_result = result_multi(f_domain, label, classes=4, model_name="./models/f_domain-4-150-32-up-multiout")

    print (f"inter result: {inter_result}")
    #print (f"\n\nfreqency result: {f_result}")

