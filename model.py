
import tensorflow as tf
import numpy as np
import pandas as pd
from keras import models
from keras import layers
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Import data
data = pd.read_csv('/Users/dipakkrishnan/Desktop/avocado.csv')

def average_price_graph(label):
	average_price = data[label].mean
	plt.plot(data[label])
	plt.title("Average price of avocados for the last 3 years")
	plt.xlabel("Number of avocados")
	plt.ylabel("Avocado Price")
	plt.show()

print(data['AveragePrice'])

test_x = data['AveragePrice'][:9124]
test_y = data['AveragePrice'][:9124]
train_x = data['AveragePrice'][9125:]
train_y = data['AveragePrice'][9125:]
model = models.Sequential()
# Input - Layer
model.add(layers.Dense(50, activation = "relu", input_shape=(1, )))
# Hidden - Layers
model.add(layers.Dropout(0.3, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation = "relu"))
model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
model.add(layers.Dense(50, activation = "relu"))
# Output- Layer
model.add(layers.Dense(1, activation = "sigmoid"))
model.summary()
# compiling the model
model.compile(
 optimizer = "adam",
 loss = "binary_crossentropy",
 metrics = ["accuracy"]
)
results = model.fit(
 train_x, train_y,
 epochs= 2,
 batch_size = 500,
 validation_data = (test_x, test_y)
)
print("Test-Accuracy:", np.mean(results.history["val_acc"]))
print(model.predict(test_y))




