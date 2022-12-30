# Import Dependency
import os
from flask import Flask,  request, render_template, json, jsonify, send_from_directory
import tensorflow as tf
import numpy as np
import json
from flask_cors import CORS
import io
#import opencv Dependency
import cv2
import requests
# GCLM 
from skimage.feature import greycomatrix, graycoprops

# ----------------- calculate greycomatrix() & greycoprops() for angle 0, 45, 90, 135 ----------------------------------
def calc_glcm_all_agls(img, props, dists=[5], agls=[0, np.pi/4, np.pi/2, 3*np.pi/4], lvl=256, sym=True, norm=True):
    
    glcm = greycomatrix(img, 
                        distances=dists, 
                        angles=agls, 
                        levels=lvl,
                        symmetric=sym, 
                        normed=norm)
    feature = []
    glcm_props = [propery for name in props for propery in graycoprops(glcm, name)[0]]
    for item in glcm_props:
            feature.append(item)
    
    return feature

#------LOAD MODEL------------
# def model_load():
#     #Serve the TensorFlow.js model from the 'tensorflowjs_model' directory
    
#     global model
#     model = load.model('/Model','model.json')
#     print("Tensoflow model loaded successfully")

#----------------------------------Preprocessing---------------------------------
# def preprocess_input(img):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
#     h, w = gray.shape
#     ymin, ymax, xmin, xmax = h//3, h*2//3, w//3, w*2//3
#     crop = gray[ymin:ymax, xmin:xmax]
            
#     resize = cv2.resize(crop, (0,0), fx=0.5, fy=0.5)
#     properties = ['dissimilarity', 'correlation', 'homogeneity', 'contrast', 'ASM', 'energy']
#     wrk = calc_glcm_all_agls(resize, props=properties)
#     x = np.array(wrk, dtype=np.float32)
#     data = json.dumps({'value': x.tolist()})
    
#     return data

def preprocessing(file):
    in_memory_file = io.BytesIO()
    file.save(in_memory_file)
    data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(data, 0)
    res = cv2.resize(img, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
    # file.save("static/UPLOAD/img.png") # saving uploaded img
    # cv2.imwrite("static/UPLOAD/test.png", res) # saving processed image
    return res


#Creating WSGI Instance 
app = Flask (__name__,template_folder='templates',static_folder='statics')
CORS(app)


# #capture images from the Webcam
# cap = cv2.VideoCapture(0)

# while True:
#     #read am image from the webcam
#     ret, image = cap.read()
    
#     data = preprocess_input(image)
    
#     #Send the image to the server for prdiction
#     response = request.post('http://localhost:5000/predict', data=data)
    
#     # Get the prediction result from the response
#     prediction = response.json()

#     # Extract the class label and probability from the prediction result
#     label = prediction['label']
#     probability = prediction['probability']

#     # Display the prediction result using OpenCV
#     cv2.putText(image, f'Label: {label} ({probability:.2f})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
#     cv2.imshow('Prediction', image)

#     # Wait for a key press to exit
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break

#Release webcam
# cap.release()
    
#  ==========@APP.ROUTE===============
#------------main interface---------------
@app.route("/", methods=["GET"])
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

# ---------------prepare-----------------
@app.route("/api/prepare", methods=["POST"])
def prepare():
    file = request.files['file']
    res = preprocessing(file)
    print(res)
    return json.dumps({"image": res.tolist()})

#---------------model---------------------
@app.route('/model')
def model():
    json_data = json.load(open("Model/model.json"))
    return jsonify(json_data)

#----------------model shards----------------
@app.route('/<path:path>')
def load_shards(path):
    return send_from_directory('Model', path)



# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get the input data from the request
#     x_json = requests.form['value']

#     # Preprocess the input data
#     # Decode the input data as an image
#     x = np.array(json.loads(x_json)['value'])

#     # Run the model on the input data
#     prediction = model.predict(x)

#     # Postprocess the prediction result
#     # ...

#     # Return the prediction result
#     return prediction
    
#Run the flask application to run a server    
if __name__ == '__main__':
    app.run()