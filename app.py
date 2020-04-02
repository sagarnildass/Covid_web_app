import os
import sys
import cv2

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
import tensorflow as tf


# Some utilites
import numpy as np
from util import base64_to_pil


# Declare a flask app
app = Flask(__name__)


# You can use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.mobilenet_v2 import MobileNetV2
#from keras.applications.vgg16 import VGG16
#model = MobileNetV2(weights='imagenet')
#model = VGG16(weights='imagenet', include_top=True)

#print('Model loaded. Check http://127.0.0.1:5000/')


# Model saved with Keras model.save()
#MODEL_PATH = 'models/model-2069.data-00000-of-00001'
weightspath = 'models'
metaname = 'model.meta_eval'
ckptname = 'model-2069'

global mapping, inv_mapping

mapping = {'normal': 0, 'pneumonia': 1, 'COVID-19': 2}
inv_mapping = {0: 'normal', 1: 'pneumonia', 2: 'COVID-19'}

# Load your own trained model
#model = load_model(MODEL_PATH)
#model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')


def model_predict(image, path=weightspath, meta=metaname, ckpt=ckptname):
    
    sess = tf.Session()
    #tf.reset_default_graph()
    
    
    new_graph = tf.Graph()
    with tf.Session(graph=new_graph) as sess:
        # Import the previously export meta graph.
        saver = tf.train.import_meta_graph(os.path.join(weightspath, metaname))

        saver.restore(sess, os.path.join(weightspath, ckptname))

        graph = tf.get_default_graph()

        image_tensor = graph.get_tensor_by_name("input_1:0")
        pred_tensor = graph.get_tensor_by_name("dense_3/Softmax:0")

        #x = cv2.imread(image)
        x = np.array(image)
        x = x[:, :, ::-1].copy() 
        x = cv2.resize(x, (224, 224))
        x = x.astype('float32') / 255.0
        pred = sess.run(pred_tensor, feed_dict={image_tensor: np.expand_dims(x, axis=0)})

    print('Prediction: {}'.format(inv_mapping[pred.argmax(axis=1)[0]]))
    print(pred)
    
    if inv_mapping[pred.argmax(axis=1)[0]] == 'normal':
        pred_proba = pred[0][0]
    elif inv_mapping[pred.argmax(axis=1)[0]] == 'pneumonia':
        pred_proba = pred[0][1]
    else:
        pred_proba = pred[0][2]

    print(pred_proba)
    return inv_mapping[pred.argmax(axis=1)[0]], pred_proba



@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        #print(request.json)
        # Get the image from post request
        img = base64_to_pil(request.json)
        #print(img)
        # Save the image to ./uploads
        # img.save("./uploads/image.png")

        # Make prediction
        preds, preds_proba = model_predict(image=img)
        #print(preds)
        preds_proba = "{:.3f}".format(preds_proba) 

        result = preds.replace('_', ' ').capitalize()
        
        # Serialize the result, you can add additional fields
        return jsonify(result=result, probability=preds_proba)

    return None


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
