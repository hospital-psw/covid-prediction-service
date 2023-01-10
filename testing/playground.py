import numpy as np
import prediction_model.activations as act
import prediction_model.optimizers as opti
from prediction_model.layers import Layer
from prediction_model.model import Model, _serialize_model, _deserialize_model
from dataset.dataset import X, y

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
model.plot_loss()

_serialize_model(model, "./model.pickle")
loaded_model = _deserialize_model("./model.pickle") 
print(loaded_model.predict(X[0].reshape(1, -1)))