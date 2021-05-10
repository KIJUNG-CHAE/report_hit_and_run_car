import io
import json
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
from flask import Flask, jsonify, request


app = Flask(__name__)
np.set_printoptions(suppress=True)
model = tensorflow.keras.models.load_model('car_model_01.h5')

def car_name(predict_arr):
    id = np.argmax(predict_arr)
    
    if id == 0:
        return  "0", "spark"
    elif id == 1:
        return "1", "grand"
    elif id == 2:
        return "2", "velo"
    elif id == 3:
        return "3", "pali"
    elif id == 4:
        return "4", "starex"

def transform_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)

    return image_array


def get_prediction(image_bytes):
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image_array = transform_image(image_bytes=image_bytes)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)

    return prediction

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        img_bytes = file.read()
        predict_arr = get_prediction(image_bytes=img_bytes)
        class_id, class_name = car_name(predict_arr[0])
        return jsonify({'class_id': class_id, 'class_name': class_name})

if __name__ == '__main__':
    app.run()
