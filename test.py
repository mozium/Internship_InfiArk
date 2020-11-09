import requests
from flask import jsonify, Flask   # 需要flask主程式才能用

app = Flask(__name__)

response_error = {
    'user_info': 'Aaron',
    'model': None,
    'message': '模型名稱錯誤。',
}


@app.route('/')
def index():
    response = jsonify(response_error)
    print(type(response))
    print(response)
    return response


if __name__ == '__main__':
    app.run(port=5005)