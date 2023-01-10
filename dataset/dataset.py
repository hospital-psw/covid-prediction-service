import pandas as pd
import numpy as np

df = pd.read_csv('dataset/Covid_Dataset.csv')
df[df == 'Yes'] = 1.
df[df == 'No'] = 0.

df = df.drop(['Sanitization from Market', 'Wearing Masks'], axis=1)

y = np.array(df['COVID-19']).astype(np.float64).reshape(-1, 1)
X = np.array(df.drop(['COVID-19'], axis=1)).astype(np.float64)