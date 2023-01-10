from flask import Flask
from flask import Response
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
def predict(datapoint: np.ndarray) -> str:
    return str(predict(datapoint))


if __name__ == '__main__':
    app.run(host="localhost", port=6900, debug=True)