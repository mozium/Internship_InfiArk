import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import dash_daq as daq
import plotly.graph_objects as go   # 多圖
import plotly.express as px

import pandas as pd
import requests


# app
app = dash.Dash(
    name=__name__
)
server = app.server     # web server
app.config["suppress_callback_exceptions"] = True       # ***這個在tabs的callback需要加, 但還不知道why?


# variables
model_ids = [
    1110307018, 1110307020, 1110307021, 1110307043, 1110307046, 1110307047, 
    1110307049, 1110307050, 1110307052, 1110307053, 1110307054, 1110307055,
    1111221073, 1111221074, 1111221075, 1111528093, 1111629099, 1442036122,
    1442143111, 1442237116, 1442238117, 1442239118, 1442240119, 1442241120,
    1442342112,
    1110410036, 1110410037, 1110410042, 1110410110,
    1330620056, 1441945123
]
model_ids_options = []
for id_ in model_ids:
    model_ids_options.append({'label': f'Model: {id_}', 'value': str(id_)})


# function set
def build_banner():
    """
        目錄
    """
    return html.Div(
        id='banner',
        className='banner',
        children=[
            html.Div(
                id='banner-text',
                children=[
                    html.H5('Infiark - 企業用電量預測&異常檢測'),
                    html.H6('企業儀表板'),
                    build_button('buttons', 'login-button', 'buttons', '登入'),
                ],
            ),
            html.Div(
                id='banner-logo',
                children=[
                    html.Button(
                        id='learn-more-button',
                        children='LEARN MORE',
                        n_clicks=0,
                    ),
                    html.Img(id='logo', src=app.get_asset_url('dash-logo-new.png')),    # 之後在換圖
                ]
            )
        ]
    )


def build_tabs(nums_tabs=2, model_names=['Model-Training-Page', 'Model-Dashboard-Page']):
    """
        利用tab轉換model觀察儀表板。
    """
    # 這個可以轉換models, 這個function要改一下, **現在改
    tabs = []
    for i in range(nums_tabs):
        tab = dcc.Tab(
                id=f'models-tab-{model_names[i]}',
                label=f'{model_names[i]}',
                value=f'{model_names[i]}',
                className='custom-tab',
                selected_className='custom-tab--selected',
                )
        tabs.append(tab)

    return html.Div(
        id='tabs',
        className='tabs',
        children=[
            dcc.Tabs(
                id='app-tabs',
                value='tab2??',
                className='custom-tabs',
                children=tabs,
            )
        ]
    )


def generate_modal():
    return html.Div(
        id='markdown',
        className='modal',
        children=(
            html.Div(
                id='markdown-container',
                className='markdown-container',
                children=[
                    html.Div(
                        className='close-container',
                        children=html.Button(
                            'Close',
                            id='markdown_close',
                            n_clicks=0,
                            className='closeButton',
                        )
                    ),
                    html.Div(
                        className='markdown-text',
                        children=dcc.Markdown(
                            children=(
                                """
                                ###### 產品功能:
                                Dashboard for monotoring ...
                                
                                ###### 作者:
                                Aaron Yang

                                ###### 補充:
                                ....
                                """
                            ),
                        )
                    ),
                ]
            )
        )
    )


def build_quick_stats_panel():
    """
        製作邊panel。
    """
    return html.Div(
        id='quick-stats',
        className='row',
        children=[
            html.Div(
                id='card-1',
                children=[
                    html.P('Machine ID'),
                    daq.LEDDisplay(
                        # LED
                        id='model-led',
                        value='1110307018',
                        color='#92e0d3',
                        backgroundColor='#1e2130',
                        size=50,
                    ),
                ],
            ),
            html.Div(
                id='card-2',
                children=[
                    html.P('Gauge: 預測準度顏色變化'),
                    daq.Gauge(
                        id='progress-gauge',
                        max=100,
                        min=0,
                        showCurrentValue=True,  # default size 200 pixel
                    ),
                ],
            ),
            html.Div(
                id='utility-card',
                children=[
                    daq.StopButton(
                        id='stop-button',
                        size=160,
                        n_clicks=0
                    )
                ]
            )
        ]
    )


def generate_section_banner(title):
    return html.Div(className='section-banner', children=title)


def build_chart_panel(model_id):
    """
        建立趨勢圖, func還沒設計好。
    """
    # *** 這邊可以改變圖的style設計
    data = get_data_from_api(id_=model_id)
    X = list(range(len(data['y_real'])))

    fig = go.Figure(
        layout=go.Layout(title=go.layout.Title(text='kw_kwh')),
    )
    fig.add_trace(go.Scatter(x=X, y=data['y_pred'], mode='lines', name='pred'))
    fig.add_trace(go.Scatter(x=X, y=data['y_real'], mode='lines', name='real'))
    return html.Div(
        id='control-trends-container',
        className='twelve columns',
        children=[
            generate_section_banner('趨勢圖'),
            # ***important graph
            dcc.Graph(
                id='control-trends-live',
                figure=fig,
            )
        ]
    )


def build_button(div_id, id_, class_name, bt_name):
    """
        建立按鈕
    """
    button = html.Button(
        id=id_,
        className=class_name,
        title='Click it!',
        children=bt_name,
    )
    return html.Div(
        id=div_id,
        className=class_name + '_div',
        children=button,
    )


def generate_piechart(id_):
    return dcc.Graph(
        id=id_,
        figure={
            'data': [
                {
                    'labels': [],
                    'values': [],
                    'type': 'pie',
                    'marker': {'line': {'color': 'white', 'width': 1}},
                    'hoverinfo': 'label',
                    'textinfo': 'label',
                }
            ],
            'layout': {
                'margin': dict(l=20, r=20, t=20, b=20),
                'showlegend': True,
                'paper_bgcolor': 'rgba(0,0,0,0)',
                'plot_bgcolor': 'rgba(0,0,0,0)',
                'font': {'color': 'white'},
                'autosize': True,
            },
        },
    )


def generate_dropdown(div_id, className, options, searchable, placeholder):
    dropdown = dcc.Dropdown(
        id=div_id,
        options=options,
        searchable=searchable,
        value=str(options[0]['value']),
        placeholder=placeholder,
    )
    return html.Div(
        id='div-' + div_id,
        className=className,
        children=dropdown,
    )


# def generate_graph_trend(model_id):
#     # ***
    
#     fig = go.Figure(
#         layout=go.Layout(title=go.layout.Title(text='mape: ?')),
#     )
#     fig.add_trace(go.Scatter(x=X, y=list(map(lambda x: x**2, X)), mode='lines', name='pred'))
#     fig.add_trace(go.Scatter(x=X, y=list(map(lambda x: x**2+1, X)), mode='lines', name='real'))
#     graph = dcc.Graph(
#         id='graph-trends',
#         figure=fig,
#     )
#     slider = dcc.Slider(
#         id='time-slider',
#         min=0,
#         max=1000,
#         value=168,
#         step=1
#     )
#     return html.Div(
#         id='div-graph-trends',
#         children=[graph, slider]
#     )


def get_data_from_api(url='http://localhost:8000/model/gru/', id_='1110307018'):
    resp = requests.get(url + str(id_))
    resp = resp.json()['data']
    return resp


# app layout
app.layout = html.Div(
    id='big-app-container',
    children=[
        build_banner(),
        html.Div(
            id='app-container',
            children=[
                build_tabs(),
            ]
        ),
        generate_dropdown('model-dropdown', 'model-dropdown', model_ids_options, True, 'Search a model id'),
        # ?
        html.Div(
            id='app-content',
            children=html.Iframe(
                src='http://localhost:8003',
                width=700,
                height=800,
            ),
        ),
        generate_modal(),
        generate_piechart('pie-chart'),
    ]
)


# callbacks function
@app.callback(
    Output('app-content', 'children'),
    [Input('app-tabs', 'value'), Input('model-dropdown', 'value')],
)
def render_tab_content(tab_switch, model_id):
    if tab_switch == 'Model-Training-Page':
        return html.Iframe(
            src='http://localhost:8003',
            width=700,
            height=800,
        )
    return html.Div(
        id='status-container',
        children=[
            build_quick_stats_panel(),
            html.Div(
                id='graphs-container',
                #children=list('之後補'),
                children=[build_chart_panel(model_id)]
            ),
            
        ],
    )


@app.callback(
    Output('markdown', 'style'),
    [Input('learn-more-button', 'n_clicks'), 
     Input('markdown_close', 'n_clicks')],
)
def update_click_output(button_click, close_click):
    ctx = dash.callback_context

    if ctx.triggered:
        prop_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if prop_id == 'learn-more-button':
            return {'display': 'block'}
    return {'display': 'none'}


@app.callback(
    Output('model-led', 'value'),
    [Input('model-dropdown', 'value')]
)
def update_model_change(model_id):
    # ***這個要慢慢改良。
    # 缺畫面render, 應該是搭配update_tab_content去做chain effect
    if not model_id:
        return '9999999999'
    return model_id

# Running the server
if __name__ == '__main__':
    app.run_server(debug=True, port=8050)