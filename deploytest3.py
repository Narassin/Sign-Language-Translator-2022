# Import Depenency (for upload)
from flask_cors import CORS
from flask import Flask, request, render_template, json, jsonify, send_from_directory
import json
import cv2
import numpy as np
import io
from skimage.feature import greycomatrix, greycoprops



# Import Dependency
# import os
# from flask import Flask,  request, render_template, json, jsonify, send_from_directory
# import tensorflow as tf
# import numpy as np
# import json
# from flask_cors import CORS
# import io
#import opencv Dependency
import cv2
import requests

#Creating WSGI Instance 
app = Flask (__name__,template_folder='templates',static_folder='statics')
CORS(app)

#  ==========@APP.ROUTE===============
#------------main interface---------------
@app.route("/" )
def index():
    return render_template('index.html')

@app.route("/realtime", methods=['GET', 'POST'])
def wbcam():
    return render_template('wbcam.html')

@app.route("/upload",methods=["GET"])
def upload():
    return render_template('upload.html')

@app.route("/about")
def about():
    return render_template('about.html')

# ---------------prepare-----------------
@app.route("/api/prepare", methods=["POST"])
def prepare():
    file = request.files['file']
    res = preprocessing(file)
    return json.dumps({"image": res})

#---------------Load Model--------------------
@app.route('/model')
def model():
    json_data = json.load(open("./model_js/model.json"))
    return jsonify(json_data)

#----------------model shards----------------
@app.route('/<path:path>')
def load_shards(path):
    return send_from_directory('model_js', path)


# ------------- Pre Processing -------------------------------------------
def calc_glcm_all_agls(img, props, dists=[5], agls=[0, np.pi/4, np.pi/2, 3*np.pi/4], lvl=256, sym=True, norm=True):
    
    glcm = greycomatrix(img, 
                        distances=dists, 
                        angles=agls, 
                        levels=lvl,
                        symmetric=sym, 
                        normed=norm)
    feature = []
    glcm_props = [propery for name in props for propery in greycoprops(glcm, name)[0]]
    for item in glcm_props:
            feature.append(item)
    df = feature
    return df

def preprocessing(file):
    in_memory_file = io.BytesIO()
    file.save(in_memory_file)
    data = np.frombuffer(in_memory_file.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(data, 0)
    properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']
    resize = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
    
    res = calc_glcm_all_agls(resize,props=properties)
    # file.save("static/UPLOAD/img.png") # saving uploaded img
    # cv2.imwrite("static/UPLOAD/test.png", res) # saving processed image
    
    return res



#------LOAD MODEL------------
# def model_load():
#     #Serve the TensorFlow.js model from the 'tensorflowjs_model' directory
    
#     global model
#     model = tf.loadLayersModel('/Model','model.json')
#     print("Tensoflow model loaded successfully")


    
#==============Run the flask application to run a server==================    
if __name__ == "__main__":
    app.run()