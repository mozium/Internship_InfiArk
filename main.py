from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import pandas as pd

app = Flask(__name__)
db = SQLAlchemy()

# mysql connection setting
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# app.config['SQLALCHEMY_DATABASE_URI'] = "mysql+pymysql://user_name:password@IP:3306/db_name"

db.init_app(app)

@app.route('/')
class user(db.Model):
    __tablename__ = 'user'
    u_id = db.Column(db.Integer, primary_key=True)
    u_acc = db.Column(db.String(30), unique=True, nullable=False)
    u_pwd = db.Column(db.Integer, nullable=False)
    u_type = db.Column(db.Integer, nullable=False)

    def __init__(self, u_id, u_acc, u_pwd, u_type):
        self.u_id = u_id
        self.u_acc = u_acc
        self.u_pwd = u_pwd
        self.u_type = u_type


@app.route('/')
def index():
    # Create data
    db.create_all()

    return 'ok'