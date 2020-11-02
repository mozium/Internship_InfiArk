import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, GRU, SimpleRNN
from tensorflow.keras.layers import concatenate, add
from tensorflow.keras.layers import Dropout, BatchNormalization, Input, Embedding
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error



def pre_process(x_data, y_data, random_state: int=42, 
                standard_list: tuple=(0, 168), scaler: str='Standard', 
                pred_step: int=24, shuffle: bool=False):
    """
        資料前處理：資料切分、標準化
    """
    

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25, random_state=random_state, shuffle=shuffle)
    # 前168的是可以標準化的數值
    
    if scaler == 'MinMax':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()
    s, e = standard_list

    # 轉成 ndarray
    x_train, x_test = x_train.values, x_test.values
    y_train, y_test = y_train.values, y_test.values

    x_train[:, s:e] = scaler.fit_transform(x_train[:, s:e])
    x_test[:, s:e] = scaler.fit_transform(x_test[:, s:e])

    # x_test, y_test因為24小時預測一次, 所以資料要切分, 就不會那麼多

    return x_train, x_test[::24], y_train, y_test[::24]

def get_predict_data(x_test, time_step=24, is_rnn=False):
    # x_test: (num_datapoints, 197) or x_test_rnn: (num_datapoints, 197, 1)
    """
        預測是每24小時, 要取最新的作為預測值, 故每一次預測的前24筆是最新的
    """
    if not is_rnn:
        totals = x_test.shape[0] // time_step * time_step
        return x_test[0:totals:24]
    else:
        totals = x_test[0].shape[0] // time_step * time_step
        return x_test[0][0:totals:24], x_test[1][0:totals:24]

def get_real_pred(y_test, time_step=24, is_rnn=False):
    return y_test[:, 0:time_step].flatten()


def MAPE(y_test, y_pred):
    assert len(y_test) == len(y_pred)
    total = 0
    cumsum = 0
    for e1, e2 in zip(y_pred, y_test):
        if e2 == 0:
            continue
        else:
            cumsum += abs((e1 - e2) / e2)
            total += 1
    return round(cumsum / total * 100, 3)


def predict(id_=None, model=None, is_rnn=True):
    x = pd.read_csv(f'./datasets/x_{id_}.csv')
    y = pd.read_csv(f'./datasets/y_{id_}.csv')

    x = x.drop(columns=['id', 'weekday', 'hr'])
    y = y.drop(columns=['id'])

    # 前處理, 標準化

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42, shuffle=False)

    # scaler = StandardScaler()
    scaler = MinMaxScaler()
    s, e = 0, 168

    # 轉成 ndarray
    x_train, x_test = x_train.values, x_test.values
    y_train, y_test = y_train.values, y_test.values

    x_train[:, s:e] = scaler.fit_transform(x_train[:, s:e])
    x_test[:, s:e] = scaler.fit_transform(x_test[:, s:e])

    x_train = x_train[:, 0:197]
    x_test = x_test[:, 0:197][:1500]    # 發現有一些怪怪的

    y_train = y_train[:, 0:48]
    y_test = y_test[:, 0:48][:1500]

    x_train_rnn = x_train[:, 0:168]
    x_test_rnn = x_test[:, 0:168]

    x_train_rnn = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test_rnn = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


    y_real = get_real_pred(get_predict_data(y_test))
    y = get_real_pred(get_predict_data(y.values))
    if not is_rnn:
        y_pred = get_real_pred(model.predict(get_predict_data(x_test)))
    else:
        y_pred = get_real_pred(model.predict([get_predict_data(x_test_rnn[:, 168:]), get_predict_data(x_test_rnn[:, 0:168])]))
    mape = MAPE(y_real, y_pred)
    mse = mean_squared_error(y_true=y_real, y_pred=y_pred).numpy()
    mae = mean_absolute_error(y_true=y_real, y_pred=y_pred).numpy()
    
    
    return y_real, y_pred, mape, mse, mae, y






if __name__ == '__main__':
    pass