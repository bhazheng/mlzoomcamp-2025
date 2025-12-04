import os
import urllib.request
import numpy as np
import onnxruntime as ort
from PIL import Image

MODEL_NAME = os.getenv("MODEL_NAME", "hair_classifier_empty.onnx")

session = ort.InferenceSession(MODEL_NAME)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

def download_image(url):
    with urllib.request.urlopen(url) as resp:
        buffer = resp.read()
    stream = urllib.request.io.BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size=(200, 200)):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.BILINEAR)
    return img

def preprocess(img):
    x = np.array(img, dtype='float32')
    x = x / 255.0
    
    x = x.transpose(2, 0, 1)

    x = np.expand_dims(x, axis=0)
    return x

def predict(url):
    img = download_image(url)
    img = prepare_image(img)
    X = preprocess(img)
    
    result = session.run([output_name], {input_name: X})
    return result[0][0][0]

def lambda_handler(event, context):
    url = event.get('url')
    prediction = predict(url)
    return {'prediction': float(prediction)}