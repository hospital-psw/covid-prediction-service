from flask import Flask
from flask import Response
from flask import request

from data_model.symptoms import Symptoms
from prediction_model.model_manipulation import train_and_save_model
from prediction_model.model_manipulation import predict as model_predict


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
    json_dict = request.get_json(force=True)
    symptoms = Symptoms.load_from_json_dict(json_dict)
    vector = symptoms.to_numpy()
    prediction, confidence = model_predict(vector)
    print(prediction, confidence)

    return Response(status=200)


if __name__ == '__main__':
    app.run(host="localhost", port=6900, debug=True)