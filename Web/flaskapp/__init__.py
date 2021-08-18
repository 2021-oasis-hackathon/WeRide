#flask/__init__.py 
from flask import Flask, g, request, Response, make_response, session, render_template
from flask import Markup, redirect, url_for
from datetime import datetime, date, timedelta
from werkzeug.utils import secure_filename
import os
import sys
#sys.path.append('C:\\Users\\user\\Desktop\\오아시스\\WeRide\\yolov5')
#from . import main, total_score


app=Flask(__name__)
app.debug = True
#app.jinja_env.trim_blocks = True

@app.route("/success")
def success():
    cmd=("python ../yolov5/main.py --source ../test_video/2.mp4")
    os.system(cmd)
    return render_template('success.html')


@app.route("/loading")
def loading():
    
    return render_template('loading.html')

@app.route('/analysis')
def analysis():
    f=open("C:/Users/user/Desktop/오아시스/WeRide/total_score.txt",'r')
    f2=open("C:/Users/user/Desktop/오아시스/WeRide/table_score.txt", 'r')
    total_score=f.readline()
    table_score=f2.readlines()
    f.close()
    f2.close()
    return render_template('analysis.html', total=total_score, table=table_score)

@app.route('/score')
def score():
    f=open("C:/Users/user/Desktop/오아시스/WeRide/total_score.txt",'r')
    total_score=f.readline()
    f.close()
    return render_template("score.html", total=total_score)

@app.route('/service')
def service():
    return render_template("service.html")

@app.route('/upload')
def render_file():
    return render_template('upload.html')

@app.route('/fileUpload', methods=['GET', 'POST'])
def upload_file():
    loading()
    if request.method == 'POST':
        loading()
        f=request.files['file'] 
        filename=f.filename
        #dirname=f.filename[:4]
        f.save('../test_video/'+secure_filename(filename)) #저장할 경로 + 파일명
        

        return render_template('loading.html', file=filename)

@app.route('/servicepage')
def servicepage():
    return render_template("servicepage.html")

@app.route('/')
def main1():
    return render_template("mainpage.html")









