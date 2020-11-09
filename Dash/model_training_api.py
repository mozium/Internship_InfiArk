"""
程式說明:
    供 Dash 訓練頁面呼叫, 傳遞json格式, 透過解析去排程訓練, 回傳對應資訊。
"""

import json
# import tensorflow as tf
import numpy as np
import pandas as pd

from flask import Flask, jsonify, request
from models import get_dnn, get_gru, get_lstm, train_model, predict

# web server
app = Flask(__name__)
app.config['DEBUG'] = True


# API
@app.route('/model_training/', methods=['POST'])
def model_training():
    # 取得json並解析
    data_json = request.get_json()

    user_info = data_json.get('user_info', default='aaron')
    x_train = data_json.get('x_train')
    y_train = data_json.get('y_train')
    real_data = data_json.get('real_data')      # real y
    model_str = data_json.get('model_str', default='dnn')
    loss = data_json.get('loss', default='mse')
    is_shuffle = data_json.get('is_shuffle', default=True)
    is_rnn = True

    # 或許可以再包一層 Model(model_str)...
    if model_str == 'dnn':
        model = get_dnn(loss=loss)
        is_rnn = False
    elif model_str == 'gru':
        model = get_gru(loss=loss)
    elif model_str == 'lstm':
        model = get_lstm(loss=loss)
    else:
        # 回傳json, 但表示沒東西
        response_error = {
            'user_info': user_info,
            'model_address': None,
            'message': '模型名稱錯誤。',
        }
        return jsonify(response_error)

    model, history = train_model(model, model_str, x_train, y_train, shuffle=is_shuffle)
    # y_real, y_pred, mape, mse, mae, y = predict(id_=id_, model=model, is_rnn=is_rnn)
    # *****儲存在雲端, 要個人化空間, 這怎麼搞!?

    
    


if __name__ == '__main__':
    app.run(debug=True, port=8111)