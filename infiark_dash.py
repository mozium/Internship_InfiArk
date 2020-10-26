# 流程
# 1. 讀取模型、讀取待資料(train、test)
# 2. 根據模型預測結果畫 scatter plot
    # 2.1 
# 3. callbacks ---> 
# # 可變動 Inputs: time, id_(?), models
# Outputs: 趨勢圖

import tensorflow as tf


import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go    # 多圖好像可以用這個https://plotly.com/python/time-series/
import dash_bootstrap_components as dbc


from pre_process import predict

# data & model & dash app
models = {}
model = tf.keras.models.load_model(f'./models/gru_1110307043_shuffle_400.h5')
models[1110307043] = model
y_real, y_pred, mape, y_ori = predict(id_=1110307043, model=model, is_rnn=True)

ids = [
    1110307018, 1110307020, 1110307021, 1110307043, 1110307046, 1110307047, 
    1110307049, 1110307050, 1110307052, 1110307053, 1110307054, 1110307055,
    1111221073, 1111221074, 1111221075, 1111528093, 1111629099, 1442036122,
    1442143111, 1442237116, 1442238117, 1442239118, 1442240119, 1442241120,
    1442342112,
    1110410036, 1110410037, 1110410042, 1110410110,
    1330620056, 1441945123
]

temp_ids = ids.copy()

for id_ in temp_ids:
    models[id_] = tf.keras.models.load_model(f'./models/gru_{id_}_shuffle_400.h5')
    # 一時的
    try:
        y_real, y_pred, mape, y_ori = predict(id_=id_, model=models[id_], is_rnn=True)
        print(mape)
        if mape > 30:
            ids.pop(ids.index(id_))
    except:
        ids.pop(ids.index(id_))
    print(f'{id_}模型載入完畢...')

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


# layout
app.layout = html.Div([
    dcc.Dropdown(
        id='id-selector',
        options=[{'label': i, 'value': i} for i in ids],
        value=1110307043
        ),
    dcc.Graph(id='graph-trends'),
    dcc.Slider(
        id='time-slider',
        min=0,
        max=len(y_real),
        value=168,
        # marks={str(t+1): str(t+1) for t in range(len(y_real))},
        step=1
    ),
    # 新增的原始趨勢圖
    dcc.Graph(id='graph-trends-original')
])


@app.callback(
    [Output('graph-trends', 'figure'),
     Output('graph-trends-original', 'figure')],
    [Input('time-slider', 'value')]
)
def update_figure(t):
    global mape, y_ori
    d = {
        'real_val': y_real[:t],
        'pred_val': y_pred[:t],
        't': list(range(t))
    }
    d_ori = {
        'real_val': y_ori,
        't': list(range(len(y_ori)))
    }
    df = pd.DataFrame(d)
    df_ori = pd.DataFrame(d_ori)
    
    # fig = px.line(df, x='t', y='pred_val')
    fig = go.Figure(
        layout=go.Layout(
            title=go.layout.Title(text=f'mape: {mape}')
        )
    )
    fig_ori = go.Figure(
        layout=go.Layout(
            title=go.layout.Title(text='The Whole Trends')
        )
    )
    fig.add_trace(go.Scatter(x=df['t'], y=df['pred_val'], mode='lines', name='pred'))
    fig.add_trace(go.Scatter(x=df['t'], y=df['real_val'], mode='lines', name='real'))
    fig_ori.add_trace(go.Scatter(x=df_ori['t'], y=df_ori['real_val'], mode='lines'))
    return fig, fig_ori

@app.callback(
    Output('time-slider', 'value'),
    [Input('id-selector', 'value')]
)
def update_figure_byid(id_):
    global model, y_real, y_pred, models, mape, y_ori
    model = models[id_]
    y_real, y_pred, mape, y_ori = predict(id_=id_, model=model, is_rnn=True)
    return 100


if __name__ == '__main__':
    app.run_server(debug=True, port=8001)


