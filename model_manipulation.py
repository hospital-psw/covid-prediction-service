import numpy as np
import activations as act
import optimizers as opti
from layers import Layer
from model import Model, _serialize_model, _deserialize_model
from dataset import X, y


def train_and_save_model():
    global X, y

    idxs = np.arange(X.shape[0])
    np.random.shuffle(idxs)
    X = X[idxs]
    y = y[idxs]

    model = Model(X, y, int(1e3), X.shape[0])
    optimizer = opti.ADAM(1e-2, 0.9, 0.99)
    l1 = Layer(18, 8, act.ReLU(), optimizer)
    l2 = Layer(8, 1, act.Sigmoid(), optimizer)

    model.add_layer(l1)
    model.add_layer(l2)

    model.train()

    _serialize_model(model, "./model.pickle")


def predict(datapoint: np.ndarray) -> bool:
    loaded_model = _deserialize_model("./model.pickle") 
    return loaded_model.predict(datapoint) == 1
