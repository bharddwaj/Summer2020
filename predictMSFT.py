import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

msft = pd.read_csv("MSFT.csv")

print(msft.shape)
msft_returns = []
for i in range(1,1259):
    current_price = msft['Adj Close'][i] 
    prev_price = msft['Adj Close'][i-1]
    stock_returns = current_price/prev_price - 1 # (current_price - prev_price)/prev_price
    msft_returns.append(stock_returns)

msft_returns = np.array(msft_returns)

def univariate_data(start_index, end_index, history_size, target_size,dataset=msft_returns):
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

x_train, y_train = univariate_data(0,1247,1,10)

x_test, y_test = univariate_data(1247,1256,1,10)

BATCH_SIZE = 256
BUFFER_SIZE = 10000

train_univariate = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

test_univariate = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_univariate = test_univariate.batch(BATCH_SIZE).repeat()

simple_lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(8, input_shape=x_train.shape[-2:]),
    tf.keras.layers.Dense(1)
])

simple_lstm_model.compile(optimizer='adam', loss='mae')
EVALUATION_INTERVAL = 200 #epoch only runs for 200 steps as opposed to full data as is normal to save time
EPOCHS = 100

simple_lstm_model.fit(train_univariate, epochs=EPOCHS,
                      steps_per_epoch=EVALUATION_INTERVAL,
                      validation_data=test_univariate, validation_steps=50)


print(x_test.shape)
print(y_test.shape)
model_predictions = list(simple_lstm_model.predict(x_test)) #2d list [[1],[2],...] one element in each sublist
predictions = []
for i in range(len(model_predictions)):
    predictions.append(model_predictions[i][0])
print(len(predictions))
indices = list(range(len(predictions)))
print(indices)
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