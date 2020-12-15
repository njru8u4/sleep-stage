import numpy as np
from keras.layers import Conv1D  # Convolution Operation
from keras.layers import MaxPooling1D, AveragePooling1D
from keras.layers import Flatten, BatchNormalization
from keras.layers import Dense, Dropout
from keras.layers import Input, concatenate
from keras.optimizers import Adam, SGD
from keras import Model

## inter model start ##
def inter_in(input1):
    def cell(filter_num, input, pool_size = 3):
        x = BatchNormalization()(input)
        x = Conv1D(filter_num*2, kernel_size = 2, padding = "same", activation = "relu", kernel_initializer='normal')(x)

        x1 = Conv1D(filter_num, kernel_size = 1, activation = "relu", kernel_initializer='normal')(x)

        x2 = Conv1D(filter_num, kernel_size = 1, padding = "same", activation = "relu", kernel_initializer='normal')(x)
        x2 = Conv1D(filter_num*4, kernel_size = 5, padding = "same", activation = "relu", kernel_initializer='normal')(x2)

        x3 = Conv1D(filter_num*4, kernel_size = 2, padding = "same", activation = "relu", kernel_initializer='normal')(x)
        x3 = Conv1D(filter_num, kernel_size = 1, padding = "same", activation = "relu", kernel_initializer='normal')(x3)

        x4 = Conv1D(filter_num*4, kernel_size = 3, padding = "same", activation = "relu", kernel_initializer='normal')(x)
        x4 = Conv1D(filter_num, kernel_size = 1, activation = "relu", kernel_initializer='normal')(x4)

        merge = concatenate([x1, x2, x3, x4], axis=2)
        pooling = MaxPooling1D(pool_size = pool_size)(merge)
        return pooling
        
    x = input1
    cell1 = cell(16, x, 2)
    cell1 = cell(16, cell1, 2)
    cell1 = cell(16, cell1, 2)

    return cell1

def inter_train_model(input_data, label, classes):
        
    input1 = Input(shape=(len(input_data[0]), 1), name="input_1")
    M = inter_in(input1)
    flat = Flatten()(M)
    drop = Dropout(0.5)(flat)
    D = Dense(units=64, activation = "relu", name = "dense_out")(drop)
    D1 = Dense(units = classes, activation = "softmax", name = "output_layer")(D)

    model = Model(inputs= input1, outputs = D1)
    model.compile(optimizer = Adam(lr = 0.0005), loss = 'categorical_crossentropy', metrics=["categorical_accuracy"])
    model.summary()
    from keras.callbacks import EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=2)
    history = model.fit(input_data, label, epochs=100, validation_split= 0.3, callbacks=[early_stopping], batch_size=32)
    return model
## inter model end ##


## f start ##
def f_in(input1):
    def cell(filter_num, input, pool_size = 3):
        x = input
        x = Conv1D(filter_num*2, kernel_size = 2, padding = "same", activation = "relu", kernel_initializer='normal')(x)
        #x = BatchNormalization()(x)
        x1 = Conv1D(filter_num, kernel_size = 1, activation = "relu", kernel_initializer='normal')(x)

        x2 = Conv1D(filter_num*4, kernel_size = 5, padding = "same", activation = "relu", kernel_initializer='normal')(x)
        x2 = Conv1D(filter_num, kernel_size = 1, padding = "same", activation = "relu", kernel_initializer='normal')(x2)

        x3 = Conv1D(filter_num*4, kernel_size = 2, padding = "same", activation = "relu", kernel_initializer='normal')(x)
        x3 = Conv1D(filter_num, kernel_size = 1, padding = "same", activation = "relu", kernel_initializer='normal')(x3)

        x4 = Conv1D(filter_num*4, kernel_size = 3, padding = "same", activation = "relu", kernel_initializer='normal')(x)
        x4 = Conv1D(filter_num, kernel_size = 1, activation = "relu", kernel_initializer='normal')(x4)

        merge = concatenate([x1, x2, x3, x4], axis=2)
        merge = BatchNormalization()(merge)
        pooling = MaxPooling1D(pool_size = pool_size, strides=1)(merge)
        return merge
        
    x = input1
    cell1 = cell(16, x, 2)
    #cell1 = MaxPooling1D(pool_size=2)(cell1)
    cell1 = cell(16, cell1, 2)
    #cell1 = cell(16, cell1, 2)
    #cell1 = MaxPooling1D(pool_size=2)(cell1)
    cell1 = cell(32, cell1, 2)
    #cell1 = cell(32, cell1, 2)
    
    return cell1

def frequency_train_model(input_data, label, classes):
        
    input1 = Input(shape=(len(input_data[0]), 1), name="input_1")
    M = f_in(input1)
    flat = Flatten()(M)
    #drop = Dropout(0.2)(flat)
    drop = BatchNormalization()(flat)
    D = Dense(units=32, activation = "relu", name = "dense_out")(flat)
    D1 = Dense(units = classes, activation = "softmax", kernel_initializer='normal', name = "output_layer")(D)

    model = Model(inputs= input1, outputs = D1)
    model.compile(optimizer = Adam(lr = 0.0005), loss = 'categorical_crossentropy', metrics=["categorical_accuracy"])
    model.summary()
    from keras.callbacks import EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=2)
    history = model.fit(input_data, label, epochs=100, validation_split= 0.3, callbacks=[early_stopping], batch_size=64, shuffle=True)
    return model
## f model end ##


## multi start ##
def multi_output(input_data, label, classes):
    def cell(filter_num, input, pool_size = 3):
        x = input
        #x = BatchNormalization()(input)
        x = Conv1D(filter_num*2, kernel_size = 2, padding = "same", activation = "relu", kernel_initializer='normal')(x)

        x1 = Conv1D(filter_num, kernel_size = 1, activation = "relu", kernel_initializer='normal')(x)

        x2 = Conv1D(filter_num, kernel_size = 1, padding = "same", activation = "relu", kernel_initializer='normal')(x)
        x2 = Conv1D(filter_num*4, kernel_size = 5, padding = "same", activation = "relu", kernel_initializer='normal')(x2)

        x3 = Conv1D(filter_num*4, kernel_size = 2, padding = "same", activation = "relu", kernel_initializer='normal')(x)
        x3 = Conv1D(filter_num, kernel_size = 1, padding = "same", activation = "relu", kernel_initializer='normal')(x3)

        x4 = Conv1D(filter_num*4, kernel_size = 3, padding = "same", activation = "relu", kernel_initializer='normal')(x)
        x4 = Conv1D(filter_num, kernel_size = 1, activation = "relu", kernel_initializer='normal')(x4)

        merge = concatenate([x1, x2, x3, x4], axis=2)
        pooling = MaxPooling1D(pool_size = pool_size)(merge)
        return merge

    def output(pooling):
        flat = Flatten()(pooling)
        drop = Dropout(0.5)(flat)
        D = Dense(units=64, activation="relu")(flat)
        D1 = Dense(units=classes, activation="softmax")(D)
        return D1
    

    input1 = Input(shape=(len(input_data[0]), 1), name="input_1")

    x = input1
    cell1 = cell(16, x, 2)
    #output1 = output(cell1)
    #cell1 = MaxPooling1D(pool_size=2)(cell1)
    cell2 = cell(16, cell1, 2)
    #output2 = output(cell2)
    cell3 = cell(16, cell2, 2)
    output3 = output(cell3)
    cell3 = MaxPooling1D(pool_size=2)(cell3)
    cell4 = cell(16, cell3, 2)
    cell5 = cell(16, cell4, 2)
    flat = Flatten()(cell5)
    drop = Dropout(0.5)(flat)
    D = Dense(units=32, activation="relu", name = "dense_out")(flat)
    D1 = Dense(units=classes, activation="softmax", name="output_layer")(D)
    
    model = Model(inputs= input1, outputs = [output3, D1])
    model.compile(optimizer = Adam(lr = 0.001), loss = 'categorical_crossentropy', metrics=["categorical_accuracy"])
    model.summary()
    from keras.callbacks import EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=2)
    history = model.fit(input_data, [label, label], epochs=100, validation_split= 0.3, callbacks=[early_stopping], batch_size=128)
    return model
    


