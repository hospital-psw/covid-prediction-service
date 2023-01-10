from flask import Flask
from flask import Response
from flask import request
import numpy as np

from prediction_model.model_manipulation import train_and_save_model, predict

app = Flask(__name__)

@app.get('/')
def hello_world():
    return "<p>Hellooooo</p>"


@app.get('/model/train')
def train_model():
    train_and_save_model()
    return Response(status=200)


@app.post('/model/predict')
def predict() -> str:
    print(request.get_json(force=True))
    return Response(status=200)


if __name__ == '__main__':
    app.run(host="localhost", port=6900, debug=True)