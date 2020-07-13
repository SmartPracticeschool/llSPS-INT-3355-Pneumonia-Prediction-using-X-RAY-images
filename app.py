# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 15:11:48 2020

@author: aarya singh
"""

import numpy as np
import os
from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
global graph
graph = tf.compat.v1.get_default_graph()
from flask import Flask , request, render_template



tf.compat.v1.enable_eager_execution()

app = Flask(__name__)
model = load_model("P_model.h5")

@app.route('/')
def index():
    return render_template('base.html')

@app.route('/predict',methods = ['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        filepath = os.path.join(basepath,'uploads',f.filename)
        print("upload folder is ", filepath)
        f.save(filepath)
        
        img = image.load_img(filepath,target_size = (128,128))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis =0)
        
        tf.compat.v1.enable_eager_execution()
        
        #with tf.Graph().as_default():
        #model.compile()
        #model.run_eagerly = True
        preds = model.predict_classes(x)
       
        if (preds == 0):
            a = 'Normal. Congaratulation!'
        else:
            a = 'Pneumonic.'
        
       
        text = "The patient's condition is " + a   
        
    return text
if __name__ == '__main__':
    app.run(debug = True, threaded = False)