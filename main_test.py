import numpy as np
import pandas as pd
import activations as act
import optimizers as opti
from model import Model
from layers import Layer


df = pd.read_csv("dataset.csv")

# drop needless
df = df.drop(["cats", "airport"], axis=1)

# fill nans with means
for col in df.columns:
    if col in ['label', 'city']:
        continue
    df[col] = df[col].fillna(value=df[col].mean())

# OHE cities
cities = {'Beograd': 0, 'Novi Sad': 1, 'Nis': 2, 'Kragujevac': 3}
X = df.drop('label', axis=1)
for city in cities.keys():
    X['city'] = X['city'].replace(city, cities[city])
X = X.to_numpy(dtype=np.object_)

temp = np.zeros((X.shape[0], len(np.unique(X[:, 0]))))
for val in np.unique(X[:, 0]):
    vec = np.zeros(len(np.unique(X[:, 0])))
    vec[int(val)] = 1
    temp[X[:, 0] == val] = vec

X = np.concatenate([X[:, 1:], temp], axis=1)
X = np.array(X, dtype=np.float_)

# normalize
X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))

# string labels to value labels
y = df['label']
y = y.to_numpy(np.object_)
y = np.where(y == y[0], 0, 1)
y = y.reshape((y.shape[0], 1))

# get tests
idx = np.random.randint(0, X.shape[0]-1, 1000)
mask = np.ones(X.shape[0], bool)
mask[idx] = False
Xt = X[~mask]
yt = y[~mask]
X = X[mask]
y = y[mask]

# model stuff
model = Model(X, y, int(1e2), 16)
optimizer = opti.ADAM(1e-1, 0.9, 0.99)
l1 = Layer(9, 5, act.Tanh(), optimizer)
l2 = Layer(5, 1, act.Sigmoid(), optimizer)

model.add_layer(l1)
model.add_layer(l2)

model.train()
model.plot_loss()
model.score(Xt, yt)