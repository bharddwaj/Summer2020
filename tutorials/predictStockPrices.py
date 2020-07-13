# tutorial link
# https://www.kdnuggets.com/2018/11/keras-long-short-term-memory-lstm-model-predict-stock-prices.html

# data link
# https://github.com/mwitiderrick/stockprice
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

dataset_train = pd.read_csv("stockprice-master/NSE-TATAGLOBAL.csv")
training_set = dataset_train.iloc[:,1:2].values #.iloc then we do 1:3 and loc uses words
training_set2 = dataset_train.iloc[:,1:3].values 
# print(dataset_train.head())
# print(training_set)
# print(training_set2)


sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# print(dataset_train.tail())
X_train = []
y_train = []
for i in range(60, 2035):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

print(dataset_train.shape)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1],1)))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.LSTM(units = 50, return_sequences = True))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.LSTM(units = 50, return_sequences = True))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.LSTM(units = 50))
model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Dense(units = 1))

model.compile(optimizer = 'adam', loss= 'mse')
model.fit(X_train, y_train, epochs = 50)

dataset_test = pd.read_csv("stockprice-master/tatatest.csv")
real_stock_prices = dataset_test.iloc[:, 1:2].values

dataset_total = pd.concat((dataset_train["Open"], dataset_test["Open"]), axis = 0)
# print(len(dataset_total))
# print(len(dataset_test))
# print(len(dataset_train))

inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
# print(inputs.shape)
inputs = inputs.reshape(-1,1)
# print(inputs.shape)
inputs = sc.transform(inputs) #just transform as opposed to fit_transform

X_test = []
for i in range(60,76):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price) #undo the transform ?

plt.plot(real_stock_prices, color = 'black', label = 'TATA Stock Price')
plt.plot(predicted_stock_price, color = 'green', label = 'Predicted TATA Stock Price')
plt.title('TATA Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('TATA Stock Price')
plt.legend()
plt.show()