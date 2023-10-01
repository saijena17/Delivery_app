import os
from PIL import Image
import base64
import io
from io import BytesIO
import numpy as np
from flask import Flask, request, render_template
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
import matplotlib.pyplot as plt
#from werkzeug.utils import secure_filename
from werkzeug.datastructures import ImmutableMultiDict
from wtforms.validators import InputRequired
from flask import jsonify
import skimage
import cv2
import ssl
import numpy as np
import pandas as pd
import keras
from keras import optimizers
import tensorflow as tf
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from keras.layers import Input, Conv2D, Dense, concatenate
from tensorflow.keras.models import load_model
from keras.models import Model
import requests
import math
import cv2 as cv
from flask import Flask, render_template, Response

delivery_map = {'Dunzo': 0, 'Others': 1, 'Swiggy': 2, 'Ubereats': 3, 'Zomato': 4}
inv_delivery_map = {0:'Dunzo', 1:'Others', 2:'Swiggy', 3:'Ubereats', 4:'Zomato'}
model = load_model("./delivery_model.h5")

classes = ["person"]

net = cv2.dnn.readNet('./yolov3.weights', './yolov3.cfg')



#creating a flask app instance
app = Flask(__name__)

app.config["SECRET_KEY"] = "123" #form.hidden_tag() in index.html . Required with wtforms
app.config["UPLOAD_FOLDER"] = "static/files"
cap = cv2.VideoCapture(0)

#creating Class to Upload picture
class Upload(FlaskForm):
    file = FileField("File", validators=[InputRequired()]) #to upload file
    submit = SubmitField("Predict") #for submit button


@app.route('/', methods=["GET", "POST"])

@app.route('/engine', methods=["GET", "POST"]) 
def engine():
    form = Upload()
    try:
        class_ids = []
        confidences = []
        boxes = []
        if form.validate_on_submit():
            file = form.file.data #First grab the file and use it for prediction
            pic = plt.imread(file)
            pic = np.array(pic)
            net.setInput(cv2.dnn.blobFromImage(pic, 0.00392, (416,416), (0,0,0), True, crop=False))
            layer_names = net.getLayerNames()
            output_layers = ["yolo_82"]#[layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
            outs = net.forward(output_layers)
            Width = pic.shape[1]
            Height = pic.shape[0]
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.9:
                        center_x = int(detection[0] * Width)
                        center_y = int(detection[1] * Height)
                        w = int(detection[2] * Width)
                        h = int(detection[3] * Height)
                        x = center_x - w / 2
                        y = center_y - h / 2
                        class_ids.append(class_id)
                        confidences.append(float(confidence))
                        boxes.append([x, y, w, h])

            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.1)
            #check if it is people detection
            j = 0
            for i in indices:
                box = boxes[i]
                if class_ids[i]==0:
                    label = str(classes[class_id]) 
                    pic = pic[round(box[1]):round(box[1]+box[3]), round(box[0]):round(box[0]+box[2])]
                    j = 2
                    break
            if j==0:
                text = "No humans detected"
            else:
                roi_color = cv2.resize(pic, (224, 224), interpolation=cv2.INTER_LINEAR)
                roi_color = np.array([roi_color])/255
                prediction = model.predict(roi_color)
                division = inv_delivery_map[np.argmax(prediction)]
                text = "Person belongs to {}".format(division)
        else:
            text = "No humans detected"
    except:
        text = "No humans detected"
    return(render_template("./index.html", form=form, prediction_text=text)) #calling the render template along with uploading class #prediction(np.array(pic))


@app.route('/process', methods=["GET", "POST"]) #GET - to recieve file request POST - to send File or text to server
def process():
    try:
        class_ids = []
        confidences = []
        boxes = []
        image_json = request.json
        h=str(image_json["image"])
        starter = h.find(',')
        image_data = h[starter+1:]
        image_data = bytes(image_data, encoding="ascii")
        im = Image.open(BytesIO(base64.b64decode(image_data)))
        im = im.convert('RGB')
        pic = np.array(im)[1:]
        print(pic.shape)
        net.setInput(cv2.dnn.blobFromImage(pic, 0.00392, (416,416), (0,0,0), True, crop=False))
        layer_names = net.getLayerNames()
        output_layers = ["yolo_82"]
        outs = net.forward(output_layers)
        Width = pic.shape[1]
        Height = pic.shape[0]
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.9:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.1, 0.1)
        #check if is people detection
        j = 0
        for i in indices:
            box = boxes[i]
            if class_ids[i]==0:
                label = str(classes[class_id]) 
                pic = pic[round(box[1]):round(box[1]+box[3]), round(box[0]):round(box[0]+box[2])]
                j = 2
                break
        if j==0:
            text = "No humans detected"
        else:
            roi_color = cv2.resize(pic, (224, 224), interpolation=cv2.INTER_LINEAR)
            roi_color = np.array([roi_color])/255
            prediction = model.predict(roi_color)
            division = inv_delivery_map[np.argmax(prediction)]
            print(division)
            text = "Person belongs to {}".format(division)
    except:
        text="No humans detected"
    return jsonify({"division":text})
    #return(render_template("./index.html", texture=text, id = "predict_content")) 



if __name__ == '__main__':
    app.run(debug=True)