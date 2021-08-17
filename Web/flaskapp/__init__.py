#flask/__init__.py 
from flask import Flask, g, request, Response, make_response, session, render_template
from flask import Markup, redirect, url_for
from datetime import datetime, date, timedelta
from werkzeug.utils import secure_filename
import os
app=Flask(__name__)
app.debug = True
#app.jinja_env.trim_blocks = True

@app.route('/analysis')
def analysis():
    return render_template('analysis.html')

@app.route('/score')
def score():
    return render_template("score.html")

@app.route('/service')
def service():
    return render_template("service.html")

@app.route('/upload')
def render_file():
    return render_template('upload.html')

@app.route('/fileUpload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f=request.files['file'] 
        filename=f.filename
        #dirname=f.filename[:4]
        f.save('../test_video/'+secure_filename(f.filename)) #저장할 경로 + 파일명
        
        cmd=("python ../yolov5/main.py --source ../test_video/%s" %(filename))


        os.system(cmd)
        return render_template('success.html')

@app.route('/servicepage')
def servicepage():
    return render_template("servicepage.html")

@app.route('/')
def main1():
    return render_template("mainpage.html")









