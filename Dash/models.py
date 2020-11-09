"""
    目的:
        提供包裝好的模型以供初始化使用。
    選項:
        DNN, LSTM, GRU
"""

# %matplotlib inline
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import os
import time

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, GRU, SimpleRNN
from tensorflow.keras.layers import concatenate, add
from tensorflow.keras.layers import Dropout, BatchNormalization, Input, Embedding
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import mean_absolute_percentage_error

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 模型資料讀入與前處理
def get_data(x_filename, y_filename, shuffle):
    if type(x_filename) == type(''):
        x_data = pd.read_csv(x_filename)
        y_data = pd.read_csv(y_filename)

    x_data = x_data.drop(columns=['id', 'weekday', 'hr'])
    y_data = y_data.drop(columns=['id'])

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25, shuffle=shuffle)

    # 轉成 ndarray
    x_train, x_test = x_train.values, x_test.values
    y_train, y_test = y_train.values, y_test.values

    scaler = MinMaxScaler()
    s, e = 0, 168

    x_train[:, s:e] = scaler.fit_transform(x_train[:, s:e])
    x_test[:, s:e] = scaler.fit_transform(x_test[:, s:e])

    x_train = x_train[:, 0:197]
    x_test = x_test[:, 0:197]    

    y_train = y_train[:, 0:48]
    y_test = y_test[:, 0:48]

    x_train_rnn = x_train[:, 0:168]
    x_test_rnn = x_test[:, 0:168]

    x_train_rnn = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test_rnn = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    return x_train, x_test, y_train, y_test, x_train_rnn, x_test_rnn


# def get_dnn(input_shape=(168,), output_shape=48, loss='mse', optim='adam'):
#     model = Sequential()
#     model.add(Dense(64, activation='relu', input_shape=input_shape))
#     model.add(Dense(128, activation='relu'))
#     model.add(Dense(256, activation='relu'))
#     model.add(Dense(output_shape, activation='relu'))
#     model.compile(optimizer=optim, loss=loss, metrics=[loss])
    
#     return model


# def get_lstm(input_shape=(168, 1), output_shape=48, loss='mse', optim='adam'):
#     f1 = Input(shape=input_shape)
#     f2 = LSTM(units=64, return_sequences=True)(f1)
#     f2 = LSTM(units=64)(f2)
#     out = Dense(output_shape, activation='relu')(f2)

#     model = Model(inputs=[f1], outputs=[out])
#     model.compile(optimizer=optim, loss=loss, metrics=[loss])

#     return model


# def get_gru(input_shape=(168, 1), output_shape=48, loss='mse', optim='adam'):
#     f1 = Input(shape=input_shape)
#     f2 = GRU(units=64, return_sequences=True)(f1)
#     f2 = GRU(units=64, return_sequences=True)(f2)
#     f2 = GRU(units=128)(f2)
#     out = Dense(output_shape, activation='relu')(f2)

#     model = Model(inputs=[f1], outputs=[out])
#     model.compile(optimizer=optim, loss=loss, metrics=[loss])

#     return model


def get_lstm(input_shape=(168,), output_shape=48, loss='mse', optim='adam', add=False):
    if add == False:
        f1_1 = Input(shape=(29, ), name='time_input')
    else:
        f1_1 = Input(shape=(32, ), name='time_input')
    f1_2 = Input(shape=(168, 1), name='val_input')

    val_out = LSTM(units=64, return_sequences=True)(f1_2)
    val_out = LSTM(units=64)(val_out)
    time_out = Dense(32, activation='relu')((f1_1))

    z = concatenate([val_out, time_out])
    out = Dense(48, activation='relu')(z)

    lstm_2 = Model(inputs=[f1_1, f1_2], outputs=[out])
    lstm_2.compile(optimizer=optim, loss=loss, metrics=[loss])
    return lstm_2


def get_dnn(input_shape=(168,), output_shape=48, loss='mse', optim='adam', add=False):
    dnn_1 = Sequential()
    if add == False:
        dnn_1.add(Dense(units=128, activation='relu', input_shape=(197,)))
    else:
        dnn_1.add(Dense(units=128, activation='relu', input_shape=(200,)))
    dnn_1.add(Dense(units=48, activation='relu'))
    dnn_1.compile(optimizer=optim, loss=loss, metrics=[loss])

    return dnn_1

def get_gru(input_shape=(168,), output_shape=48, loss='mse', optim='adam', add=False):
    if add == False:
        f1_1 = Input(shape=(29, ), name='time_input')
    else:
        f1_1 = Input(shape=(32, ), name='time_input')
    f1_2 = Input(shape=(168, 1), name='val_input')

    val_out = GRU(units=64, return_sequences=True)(f1_2)
    val_out = GRU(units=64, return_sequences=True)(val_out)
    val_out = GRU(units=128)(val_out)
    time_out = Dense(32, activation='relu')((f1_1))
    time_out = Dense(64, activation='relu')((time_out))


    z = concatenate([val_out, time_out])
    out = Dense(48, activation='relu')(z)

    gru_3_2 = Model(inputs=[f1_1, f1_2], outputs=[out])
    gru_3_2.compile(optimizer=optim, loss=loss, metrics=[loss])

    return gru_3_2


def train_model(model, model_str, x_filename, y_filename, shuffle):
    """
        根據模型去做訓練, 最後返回模型本身以及預測結果
    """
    earlystop = EarlyStopping(monitor='val_loss',
                          patience=50,
                          verbose=1
                          )
    x_train, x_test, y_train, y_test, x_train_rnn, x_test_rnn = get_data(x_filename, y_filename, shuffle)

    if model_str == 'dnn':
        history = model.fit(x_train, y_train, 
            batch_size=512, epochs=1000, 
            validation_data=(x_test, y_test),
            callbacks=[earlystop]
            )
    elif model_str == 'lstm':
        history = model.fit([x_train_rnn[:, 168:], x_train_rnn[:, 0:168]], y_train, 
                    batch_size=512, epochs=5, 
                    validation_data=([x_test_rnn[:, 168:], x_test_rnn[:, 0:168]], y_test), 
                    callbacks=[earlystop]
                    )
    else:
        # gru
        history = model.fit([x_train_rnn[:, 168:], x_train_rnn[:, 0:168]], y_train, 
                    batch_size=512, epochs=5, 
                    validation_data=([x_test_rnn[:, 168:], x_test_rnn[:, 0:168]], y_test), 
                    callbacks=[earlystop]
                    )
    
    return model, history
            

    


if __name__ == '__main__':
    pass