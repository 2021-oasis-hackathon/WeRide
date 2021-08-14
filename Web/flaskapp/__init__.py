#flask/__init__.py

from flask import Flask, g, request, Response, make_response, session, render_template
from flask import Markup, redirect, url_for
from datetime import datetime, date, timedelta
from werkzeug.utils import secure_filename
app=Flask(__name__)
app.debug = True
#app.jinja_env.trim_blocks = True


app.config.update(
    SECRET_KEY='X1243yRH!mMwf',
    SESSION_COOKIE_NAME='pyweb_flask_session',
    PERMANENT_SESSION_LIFETIME=timedelta(31)   #31 days
)

class Nav:
    def __init__(self, title, url='#', children=[]):
        self.title=title
        self.url=url
        self.children=children

@app.route('/upload')
def render_file():
    return render_template('upload.html')

@app.route('/fileUpload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f=request.files['file'] 

        f.save('c:/Users/user/Desktop/오아시스/uploads' + secure_filename(f.filename)) #저장할 경로 + 파일명
        return 'upload 성공'


@app.route('/main1')
def main1():
    return render_template("mainpage.html")





@app.route('/wc')
def wc():
    key=request.args.get('key')
    val=request.args.get('val')
    res=Response("SET COOKIE")
    res.set_cookie(key, val)
    session['Token']='123X'
    return make_response(res)

@app.route('/rc')
def rc():
    key=request.args.get('key') #token
    val=request.cookies.get(key)
    return "cookie['" + key + "] =" + val + "," + session.get('Token')

@app.route('/delsess')
def delsess():
    if session.get('Token'):
        del session['Token']
    return "Session이 삭제되었습니다!"

@app.route('/reqenv')
def reqenv():
    return ('REQUEST_METHOD: %(REQUEST_METHOD) s <br>'
        'SCRIPT_NAME: %(SCRIPT_NAME) s <br>'
        'PATH_INFO: %(PATH_INFO) s <br>'
        'QUERY_STRING: %(QUERY_STRING) s <br>'
        'SERVER_NAME: %(SERVER_NAME) s <br>'
        'SERVER_PORT: %(SERVER_PORT) s <br>'
        'SERVER_PROTOCOL: %(SERVER_PROTOCOL) s <br>'
        'wsgi.version: %(wsgi.version) s <br>'
        'wsgi.url_scheme: %(wsgi.url_scheme) s <br>'
        'wsgi.input: %(wsgi.input) s <br>'
        'wsgi.errors: %(wsgi.errors) s <br>'
        'wsgi.multithread: %(wsgi.multithread) s <br>'
        'wsgi.multiprocess: %(wsgi.multiprocess) s <br>'
        'wsgi.run_once: %(wsgi.run_once) s') % request.environ



# request 처리 용 함수
def ymd(fmt):
    def trans(date_str):
        return datetime.strptime(date_str, fmt)
    return trans


@app.route('/dt')
def dt():
    datestr = request.values.get('date', date.today(), type=ymd('%Y-%m-%d'))
    return "우리나라 시간 형식: " + str(datestr)



#app.config['SERVER_NAME'] = 'localhost:5000'

#@app.route("/sd")
#def helloworld_local():
#    return "Hello Local.com!"


#subdomain으로 분리 가능
#@app.route("/sd", subdomain="g")
#def helloworld22():
#    return "Hello G.Local.com!!!"


@app.route('/rp')
def rp():
    q=request.args.getlist('q')
    return "q=%s" % str(q)

@app.route('/test_wsgi')
def wsgi_test():
    def application(environ, start_response):
        body = 'The request method was %s' % environ['REQUEST_METHOD']
        headers = [('Content-Type', 'text/plain'),
                    ('Content-Length', str(len(body)))]
                    
        start_response('200 OK', headers)
        return [body]
    return make_response(application)



@app.route('/res1')
def res1():
    custom_res = Response("Custom Response", 200, {'test': 'ttt'})
    return make_response(custom_res)

 
#@app.before_request
#def before_request():
#    print("before_request!!!")
#    g.str="한글"

#@app.route("/gg")
#def helloworld2():
#    return "Hello Flask World!" + getattr(g, 'str', '111')

@app.route("/")
def helloworld():
    return "HEllo Flask World!!!!!!!!!!"





