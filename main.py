import os
import numpy as np
from flask import Flask, request, render_template
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
import matplotlib.pyplot as plt
#from werkzeugutils import secure_filename
from wtforms.validators import InputRequired


import skimage
import cv2
import ssl
import numpy as np
import pandas as pd

#import keras
#from keras import optimizers
import tensorflow as tf
#from keras.models import Model
#from keras.callbacks import ReduceLROnPlateau
#from sklearn.model_selection import train_test_split
#from keras.layers import Input, Conv2D, Dense, concatenate
from tensorflow.keras.models import load_model
#from keras.models import Model

import requests
import math
import cv2 as cv
from flask import Flask, render_template, Response

delivery_map = {'Dunzo': 0, 'Others': 1, 'Swiggy': 2, 'Ubereats': 3, 'Zomato': 4}
inv_delivery_map = {0:'Dunzo', 1:'Others', 2:'Swiggy', 3:'Ubereats', 4:'Zomato'}
model = load_model("/Users/saijena/Desktop/delivery_model.h5")#('./delivery_model.h5')
upperbody_cascade_model = cv.CascadeClassifier('/Users/saijena/Desktop/Assign_images/files/upperbody.xml')
face_cascade_model = cv2.CascadeClassifier('/Users/saijena/Desktop/Assign_images/files/haarcascade_frontalface_default.xml')


#creating a flask app instance
app = Flask(__name__)

app.config["SECRET_KEY"] = "123" #orm.hidden_tag() in index.html . Required with wtforms
app.config["UPLOAD_FOLDER"] = "static/files"

#creating Class to Upload picture
class Upload(FlaskForm):
    file = FileField("File", validators=[InputRequired()]) #to upload file
    submit = SubmitField("Predict") #for submit button


@app.route('/', methods=["GET", "POST"])

@app.route('/engine', methods=["GET", "POST"]) #GET - to recieve file request POST - to send File or text to server
def engine():
    form = Upload()
    if form.validate_on_submit():
        file = form.file.data #First grab the file and use it for prediction
        pic = plt.imread(file)
        pic = np.array(pic)
        roi_color = cv2.resize(pic, (224, 224), interpolation=cv2.INTER_LINEAR)
        roi_color = np.array([roi_color])/255
        prediction = model.predict(roi_color)
        division = inv_delivery_map[np.argmax(prediction)]
        print(division)
        text = "Person belongs to {}".format(division)
        #print(prediction(np.array(pic)))
    else:
        text = "No humans detected"
    return(render_template("./index.html", form=form, prediction_text=text)) #calling the render template along with uploading class #prediction(np.array(pic))





if __name__ == '__main__':
    #for debugging
    app.run(debug=True)