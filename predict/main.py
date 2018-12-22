from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.layers import ReLU
import tensorflow as tf
import numpy as np
import cv2
from flask import Flask, request, make_response, jsonify
import os
import werkzeug
from binarize import binarize


app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

global model, graph
model = load_model('models/200.h5', custom_objects={'relu6': ReLU(6., name='relu6')})
graph = tf.get_default_graph()
UPLOAD_DIR = "/tmp"


@app.route('/', methods=['POST'])
def upload_multipart():
    if 'uploadFile' not in request.files:
        make_response(jsonify({'result': 'uploadFile is required.'}))

    file = request.files['uploadFile']
    filename = file.filename
    if '' == filename:
        make_response(jsonify({'result': 'filename must not empty.'}))

    filename = werkzeug.utils.secure_filename(filename)
    file_path = os.path.join(UPLOAD_DIR, filename)
    file.save(file_path)

    img_rows, img_cols = 400, 400
    img_channels = 1

    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    os.remove(file_path)
    binary = cv2.resize(binarize(img), (img_rows, img_cols))

    img2 = np.zeros((img_rows, img_cols, img_channels))
    img2[:, :, 0] = binary
    img2 = np.expand_dims(img2, axis=0)

    with graph.as_default():
        pred = model.predict(img2)
        pred = pred.flatten()
        # predicted_class_indices = np.argmax(pred)
        labels = ['FBMessanger', 'Instagram', 'Invalid', 'LINE', 'Twitter']
        pred_per = list(map(lambda x: x*100, pred))
        return make_response(jsonify(dict(zip(labels, pred_per))))


@app.errorhandler(werkzeug.exceptions.RequestEntityTooLarge)
def handle_over_max_file_size(error):
    print("werkzeug.exceptions.RequestEntityTooLarge")
    return 'result : file size is overed.'


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)
