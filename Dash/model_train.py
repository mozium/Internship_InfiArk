"""
        利用 Dash 框架去實作模型訓練頁面,
    其中使用dcc.load, 提供csv上傳,
    
"""
from flask import Flask, send_from_directory

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_table
import dash_daq as daq

import io
import base64
import datetime
import pandas as pd


# preprocess
from pre_process import predict

# model module
from models import get_dnn, get_lstm, get_gru, train_model


# download module
from upload_download import file_download_link


# variables
model_options = [
    {'label': 'DNN', 'value': 'dnn'},
    {'label': 'LSTM', 'value': 'lstm'},
    {'label': 'GRU', 'value': 'gru'}
]

loss_options = [
    {'label': 'MSE', 'value': 'mse'},
    {'label': 'MAE', 'value': 'mae'}
]

model_mapping = {
    'dnn': get_dnn,
    'lstm': get_lstm,
    'gru': get_gru
}

filename = None
id_ = 1110307043

# 視覺化元素建立
def build_upload(div_id, upload_id, is_out=False):
    """ 建立upload視覺化元素 """
    upload = dcc.Upload(
        id=upload_id,
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={     # 可改
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # allow multiple files to be uploaded
        multiple=True
    )
    # 把讀入資料可以render(table)在頁面上
    if is_out:
        return html.Div(
            id=div_id,
            children=[
                upload,
                html.Div(id='out-data-upload'),
            ]
        )   


def build_radio_items(options: list, id_='model-radio-select', class_='model-radio-select'):
    """ 建立RadioItems for 選擇特定事物 """
    radio_items = dcc.RadioItems(
        id=id_,
        options=options,
        value=options[0]['value'],
        labelStyle={'display': 'inline-block'}
    )
    return html.Div(
        className=class_,
        children=radio_items,
    )


def build_html_button(text='模型下載', id_='model-download-button'):
    button = html.Button(
        '模型下載',
        id=id_
    )
    return html.Div(
        [button]
    )


def parse_content(contents, filename, date):
    """ 讀進來 """
    content_type, content_string = contents.split(',')
    # print(content_type, content_string, filename, sep='\n')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # csv
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8'))    # why?
            )
        elif 'xls' in filename:
            # excel
            df = pd.read_excel(io.BytesIO(decoded))
        # *要根據user創造自己的資料夾
        df.to_csv(f'./upload/{filename}', index=False)
    except Exception as e:
        print(e)
        return html.Div([
            '處理檔案中發生錯誤。'
        ])
    return html.Div([
        html.H5(filename),
        html.H6(datetime.datetime.fromtimestamp(date)),

        # dash_table.DataTable(
        #     data=df.to_dict('records'),
        #     columns=[{'name': i, 'id': i} for i in df.columns],
        # ),

        html.Hr(),
        # 
        # html.Div('Raw Content'),
        # html.Pre(contents[0:200] + '...', style={
        #     'whiteSpace': 'pre-wrap',
        #     'wordBreak': 'break-all'
        # })
    ])


# init app
server = Flask(__name__)
app = dash.Dash(server=server, suppress_callback_exceptions=True)
app.layout = html.Div([
    build_upload('div-upload-data', 'upload-data', True),
    html.Div([
        daq.StopButton(
            id='train-button',
            label='Default',
            n_clicks=0
        ),
        html.Div(id='train-button-output'),
        file_download_link(build_html_button(), filename='dnn_6.h5'),
    ]),
    build_radio_items(options=model_options),
    build_radio_items(options=loss_options, id_='loss-radio-select', class_='loss-radio-select'),
])



# callbacks
@app.callback(
    [Output('train-button-output', 'children'),
     Output('train-button', 'label'),
     Output('train-button', 'buttonText'),
     ],
    [Input('train-button', 'n_clicks')],
    [State('model-radio-select', 'value'),      # 善用State比較合理。
     State('loss-radio-select', 'value')]
     )              
def update_model_train(n_clicks, model_str, loss_str):
    global id_
    model_func = model_mapping[model_str]
    model = model_func(loss=loss_str)
    print(model_str, n_clicks)
    if n_clicks == 0:
        return '請點擊訓練按鈕以進行訓練', '待命', f'START: {n_clicks}'
    if n_clicks > 0:
        if model_str == 'dnn':
            is_rnn= False
        else:
            is_rnn = True
        model, history = train_model(model=model, model_str=model_str, x_filename=f'./datasets/x_{id_}.csv', y_filename=f'./datasets/y_{id_}.csv', shuffle=True)
        model.save(f'./models/dash/{model_str}_{n_clicks}.h5')
        y_real, y_pred, mape, mse, mae, y = predict(id_=id_, model=model, is_rnn=is_rnn)
        
        # 模型訓練要展示的字串
        model_show_string = ''

        print('MAPE: ', mape)
        print('MSE: ', mse)
        print('MAE: ', mae)
        print('訓練結束')

        return '點擊按鈕繼續訓練新的Model', '訓練中', f'START: {n_clicks}'


@app.callback(
    [Output('download-a-link', 'hidden'),
     Output('download-a-link', 'children')],
    [Input('train-button', 'n_clicks')]
)
def update_download_button(n_clicks):
    print('trigger')
    if n_clicks != 0 and n_clicks % 2 == 1:
        return True, 'model'
    else:
        return False, 'no file'


@app.callback(
    Output('out-data-upload', 'children'),
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename'),
     State('upload-data', 'last_modified')]
)
def update_data_output(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents:
        children = [
            parse_content(c, n, d) for c, n, d in 
            zip(list_of_contents, list_of_names, list_of_dates)
        ]
        return children

if __name__ == '__main__':
    app.run_server(debug=True, port=8003)