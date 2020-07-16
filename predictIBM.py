import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

url = "https://raw.githubusercontent.com/bharddwaj/Summer2020/master/US1.IBM_190716_200715.csv"
stock = pd.read_csv(url)

print(stock.tail())

def get_returns(stock):
    '''Works specifically for the csvs that Eden gets'''
    stock_returns = []
    for i in range(1,stock.shape[0]):
        current_price = stock["<CLOSE>"][i]
        prev_price = stock["<CLOSE>"][i-1]
        stock_return = current_price/prev_price - 1 # (current_price - prev_price)/prev_price
        stock_returns.append(stock_return)
    return np.array(stock_returns)

ibm_returns = get_returns(stock)

def univariate_data(start_index, end_index, history_size, target_size,dataset):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i)
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i+target_size]) #training on predicting in the future so need labels in future
    return np.array(data), np.array(labels)

# x_train, y_train = univariate_data(0,1247,10,1)

# x_test, y_test = univariate_data(1197,1257,10,1)
BATCH_SIZE = 256
BUFFER_SIZE = 10000
x_train, y_train = univariate_data(0,87000,10,1,ibm_returns)
x_test, y_test = univariate_data(87000,88000,10,1,ibm_returns)

train_univariate = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

test_univariate = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_univariate = test_univariate.batch(BATCH_SIZE).repeat()

simple_lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(units = 8, input_shape = x_train.shape[-2:]),
    tf.keras.layers.Dense(units = 1)
    ])

def custom_loss(y_true, y_pred):
  loss = tf.keras.backend.abs(y_true - y_pred)
  loss = 10 * tf.keras.backend.mean(loss)
  return loss
simple_lstm_model.compile(optimizer='adam', loss=custom_loss, metrics = ['accuracy'])

EVALUATION_INTERVAL = 200 #epoch only runs for 200 steps as opposed to full data as is normal to save time
EPOCHS = 10

history = simple_lstm_model.fit(train_univariate, steps_per_epoch=200 ,epochs=EPOCHS)
model_predictions = list(simple_lstm_model.predict(x_test))

predictions = []
for i in range(len(model_predictions)):
    predictions.append(model_predictions[i][0])
print(len(predictions))
indices = list(range(len(predictions)))
'''
https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial
'''
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
'''
this os stuff is to fix he libiomp5.dylib error
'''
plt.plot(indices,predictions,color='red')
plt.plot(indices,y_test)
plt.show()
plt.plot(history.history["loss"])
plt.show()
