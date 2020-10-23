from flask import Flask, redirect, render_template, request, session, url_for
import pandas as pd
# import json
# import random
# import datetime
# from datetime import timedelta

# Flask Initialization
app = Flask(__name__)

# MySQL Initialization
import pymysql
from sqlalchemy import create_engine

db_reader = pymysql.connect(
    host = "127.0.0.1",
    port = int(3306),
    user = 'root',
    passwd = '',
    db = 'web',
    charset = 'utf8mb4',
    cursorclass = pymysql.cursors.DictCursor)

db_writer = create_engine('mysql+pymysql://root:666666@127.0.0.1/web')
usr_data = pd.read_sql_query("SELECT * FROM usr", db_reader)
db_cursor = db_reader.cursor()

# Session Initialization
app.config['SECRET_KEY'] = 'InfiArk_Internship_2020'


@app.route("/")
def init():
    if (session.get('usr')):
        return render_template("main.html")
    else:
        return render_template("login.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    #login functions
    if request.method == "POST":
        lusr = request.values['usr']
        lpwd = request.values['pwd']

        if lpwd == usr_data[usr_data['acc']==lusr]['pwd'].iloc[0]:
            session['usr'] = str(lusr)
            return redirect("/home")
        else:
            return redirect("/")

def register():
    #register functions
    if request.method == "POST":
        usr = request.values['rusr']
        pwd = request.values['rpwd']
        cnf = request.values['rpwd2']
        nckn = request.values['rnckn']

        if str(pwd) == str(cnf):


            session['usr'] = usr
            return redirect("/")
            '''
            user_list =  open("./schema/user.json", "r",encoding="utf-8")
            user_data = json.load(user_list)

            n = len(user_data)
            nstr = str(n)
            idstr = "p"+nstr
    
            nentry = {
                "username":user,
                "password":pwd,
                "nickname":nckn
            }

            user_data.append(nentry)

            with open(f'./schema/user.json', 'w') as json_file:
                json.dump(user_data, json_file)
                json_file.close()
            '''
    else:
        return redirect("/")


@app.route("/main.html", methods=['GET', 'POST'])
def pagerefresh():
    urls = [
        'http://www.w3schools.com',
        'http://techcrunch.com/',
        'https://www.youtube.com/',
    ]
    iframe = random.choice(urls)
    print(iframe)

    return render_template('main.html', iframe=iframe)


@app.route("/login")
def lock():
    return render_template("login.html")


@app.route("/logout")
def logout():
    session['usr'] = False
    return redirect("/")


@app.errorhandler(404) # Redirecting undefined URLs
def page_not_found(e):
    return redirect("/")


if __name__ == "__main__":
    app.run()