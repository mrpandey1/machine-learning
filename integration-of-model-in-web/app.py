from __future__ import division, print_function
import sys
import os
import glob
import re
import tensorflow as tf
import tensorflow_hub as hub
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
# Define a flask app
app = Flask(__name__)
unique_labels=['affenpinscher', 'afghan_hound', 'african_hunting_dog', 'airedale',
       'american_staffordshire_terrier', 'appenzeller',
       'australian_terrier', 'basenji', 'basset', 'beagle',
       'bedlington_terrier', 'bernese_mountain_dog',
       'black-and-tan_coonhound', 'blenheim_spaniel', 'bloodhound',
       'bluetick', 'border_collie', 'border_terrier', 'borzoi',
       'boston_bull', 'bouvier_des_flandres', 'boxer',
       'brabancon_griffon', 'briard', 'brittany_spaniel', 'bull_mastiff',
       'cairn', 'cardigan', 'chesapeake_bay_retriever', 'chihuahua',
       'chow', 'clumber', 'cocker_spaniel', 'collie',
       'curly-coated_retriever', 'dandie_dinmont', 'dhole', 'dingo',
       'doberman', 'english_foxhound', 'english_setter',
       'english_springer', 'entlebucher', 'eskimo_dog',
       'flat-coated_retriever', 'french_bulldog', 'german_shepherd',
       'german_short-haired_pointer', 'giant_schnauzer',
       'golden_retriever', 'gordon_setter', 'great_dane',
       'great_pyrenees', 'greater_swiss_mountain_dog', 'groenendael',
       'ibizan_hound', 'irish_setter', 'irish_terrier',
       'irish_water_spaniel', 'irish_wolfhound', 'italian_greyhound',
       'japanese_spaniel', 'keeshond', 'kelpie', 'kerry_blue_terrier',
       'komondor', 'kuvasz', 'labrador_retriever', 'lakeland_terrier',
       'leonberg', 'lhasa', 'malamute', 'malinois', 'maltese_dog',
       'mexican_hairless', 'miniature_pinscher', 'miniature_poodle',
       'miniature_schnauzer', 'newfoundland', 'norfolk_terrier',
       'norwegian_elkhound', 'norwich_terrier', 'old_english_sheepdog',
       'otterhound', 'papillon', 'pekinese', 'pembroke', 'pomeranian',
       'pug', 'redbone', 'rhodesian_ridgeback', 'rottweiler',
       'saint_bernard', 'saluki', 'samoyed', 'schipperke',
       'scotch_terrier', 'scottish_deerhound', 'sealyham_terrier',
       'shetland_sheepdog', 'shih-tzu', 'siberian_husky', 'silky_terrier',
       'soft-coated_wheaten_terrier', 'staffordshire_bullterrier',
       'standard_poodle', 'standard_schnauzer', 'sussex_spaniel',
       'tibetan_mastiff', 'tibetan_terrier', 'toy_poodle', 'toy_terrier',
       'vizsla', 'walker_hound', 'weimaraner', 'welsh_springer_spaniel',
       'west_highland_white_terrier', 'whippet',
       'wire-haired_fox_terrier', 'yorkshire_terrier']

def load_model(model_path):
  print(f'Loading model from :{model_path}')
  model=tf.keras.models.load_model(model_path,
                                   custom_objects={
                                       'KerasLayer':hub.KerasLayer
                                   })
  return model

def get_pred_label(prediction_probability,unique_labels):
  return unique_labels[np.argmax(prediction_probability)]  

def get_pred_get_pred(custom_preds):
    custom_pred_label=[get_pred_label(custom_preds[i]) for i in range(len(custom_preds))]
    custom_pred_label

def get_image_label(image_path,label):
  image=preprocess_image(image_path)
  return image,label

# define image size
IMG_SIZE=224
BATCH_SIZE=32

def preprocess_image(image_path,img_size=224):
  image=tf.io.read_file(image_path)
  image=tf.image.decode_jpeg(image,channels=3)
  image=tf.image.convert_image_dtype(image,tf.float32)
  image = tf.image.resize(image,size=[IMG_SIZE,IMG_SIZE])
  return image


def create_data_batches(x,batch_size=32):
    print('Creating test data branches....')
    x=[x]
    data=tf.data.Dataset.from_tensor_slices((tf.constant(x)))
    data_batch=data.map(preprocess_image).batch(BATCH_SIZE)
    return data_batch


def model_predict(custom_images_path,loaded_full_model):
    custom_data=create_data_batches(custom_images_path)
    custom_preds = loaded_full_model.predict(custom_data)
    custom_pred_label=[get_pred_label(custom_preds[i],unique_labels) for i in range(len(custom_preds))]
    return custom_pred_label


loaded_full_model=load_model('model/fullmodel.h5')
@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('./index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
                basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        # Make prediction
        result = model_predict(file_path,loaded_full_model)            
        return result[0]
    return None


if __name__ == '__main__':
    app.run(debug=True)



