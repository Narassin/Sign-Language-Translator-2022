from flask import Flask,render_template, request, url_for, redirect

# Create the web applet with the given parameter
app = Flask(__name__,template_folder='templates',static_folder='statics')

# Home Page
@app.route("/")
def index():
    return render_template('index.html')

@app.route("/realtime", methods=['GET', 'POST'])
def wbcam():
    return render_template('wbcam.html')

@app.route("/upload")
def upload():
    return render_template('upload.html')

@app.route("/about")
def about():
    return render_template('about.html')

if __name__ == "__main__":
    website_url = 'sltml.com.my:5000'
    app.config['SERVER_NAME'] = website_url
    app.run()