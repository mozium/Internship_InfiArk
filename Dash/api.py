# 此程式利用 Flask+RESTful 實現 Model API
"""
使用說明：
    外部程式利用request去取得模型預測結果。
    request利用GET去做
"""

# 深度學習框架 & 資料處理
import os
import json
import tensorflow as tf
import numpy as np
import pandas as pd
from pre_process import predict


# Server 端
from flask import Flask, jsonify, request


# data & model, 這部分其實應該搭配資料庫先做驗證model然後取得

models = {}
ids = [
    1110307018, 1110307020, 1110307021, 1110307043, 1110307046, 1110307047, 
    1110307049, 1110307050, 1110307052, 1110307053, 1110307054, 1110307055,
    1111221073, 1111221074, 1111221075, 1111528093, 1111629099, 1442036122,
    1442143111, 1442237116, 1442238117, 1442239118, 1442240119, 1442241120,
    1442342112,
    1110410036, 1110410037, 1110410042, 1110410110,
    1330620056, 1441945123
]


# web server
app = Flask(__name__)
app.config['DEBUG'] = True


# API
@app.route('/model/gru/<id_>', methods=['GET'])
def model_api(id_):
    json_name = f'gru_{id_}_shuffle_400.json'
    if json_name in os.listdir('./models/gru/json'):
        data_pass = json.load(open(f'./models/gru/json/{json_name}'))
        return jsonify({'data': data_pass})
    model = tf.keras.models.load_model(
        filepath=f'./models/gru_{id_}_shuffle_400.h5'
    )
    y_real, y_pred, mape, mse, mae, y_ori = predict(id_=id_, model=model)
    # make data jsonify to pass to client
    data_pass = {}
    data_pass['y_real'] = list(y_real)
    data_pass['y_pred'] = list(map(lambda x: float(x), y_pred))  # TypeError: Object of type float32 is not JSON serializable
    data_pass['mape'] = mape
    data_pass['mse'] = float(mse)
    data_pass['mae'] = float(mae)
    data_pass['y_ori'] = list(y_ori)
    # print(data_pass)
    
    data_json = jsonify({'data': data_pass})
    json.dump(data_pass, open(f'./models/gru/json/gru_{id_}_shuffle_400.json', 'w'))
    return  data_json


if __name__ == '__main__':
    app.run(debug=True, port=8000)