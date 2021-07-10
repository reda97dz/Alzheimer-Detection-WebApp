from flask import Flask, request, render_template
from PIL import Image
from keras.preprocessing.image import load_img
from keras.models import model_from_json
from tools.visualization.visualization import get_img_array, gradcam, make_heatmap

import os
import numpy as np
import tensorflow as tf
import cv2
import keras
import tensorflowjs as tfjs


app = Flask(__name__)

def dir_last_updated(folder):
    return str(max(os.path.getmtime(os.path.join(root_path, f))
                   for root_path, dirs, files in os.walk(folder)
                   for f in files))

@app.route("/")
def index():
    return render_template('index.html',
                            last_updated=dir_last_updated('static'))

@app.route('/predict/', methods=['POST'])
def predict():
    global model, graph
    
    f = request.files['img']
    f.save('image.png')
    
    x = get_img_array('image.png', (436,364))

    graph = tf.compat.v1.get_default_graph()

    with graph.as_default():
        print("predicting")
        # model = tf.keras.models.load_model("model\InceptionResNet-73.h5")
        json_file = open('model\model.json','r')
        model_json = json_file.read()
        json_file.close()
        model = model_from_json(model_json)
        model.load_weights("model\model.h5")
        model.compile(
            optimizer='sgd', 
            loss='categorical_crossentropy', 
            metrics=['accuracy']
        )
        prediction = model.predict(x)
        negative = prediction[0][0]
        positive = prediction[0][1]
        print(prediction)
        classIndex = np.argmax(prediction, axis=1)
        print(classIndex)
        return str(positive)
    

if __name__ == "__main__":
    app.run()
    app.run(debug=True)
