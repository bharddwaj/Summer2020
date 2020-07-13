# https://www.tensorflow.org/tutorials/structured_data/time_series
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
csv_path, _ = os.path.splitext(zip_path)

df = pd.read_csv(csv_path)

def univariate_data(dataset, start_index, end_index, history_size, target_size):
  '''
  target size is how far in the future the model needs to learn to predict
  history size is the size of the past window of information
  
  '''
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


TRAIN_SPLIT = 300000
tf.random.set_seed(13)

# First we will only use the temperature feature from the dataset

uni_data = df['T (degC)']
uni_data.index = df['Date Time']

# uni_data.plot(subplots=True)
# plt.show()

# standardize the features by subtracting mean and dividing by the standard deviation
# look into using tf.keras.utils.normalize()
uni_data = uni_data.values
uni_data = (uni_data - uni_data[:TRAIN_SPLIT].mean())/ uni_data[:TRAIN_SPLIT].std()

# Given the last 20 recorded temperature observations, the model needs to learn to
# predict the temperature at the next time step
univariate_past_history = 20
univariate_future_target = 0

x_train_uni, y_train_uni = univariate_data(uni_data, 0, TRAIN_SPLIT,
                                           univariate_past_history,
                                           univariate_future_target)

x_test_uni, y_test_uni = univariate_data(uni_data, TRAIN_SPLIT, None,
                                       univariate_past_history,
                                       univariate_future_target)

# print ('Single window of past history')
# print (x_train_uni[0])
# print ('\n Target temperature to predict')
# print (y_train_uni[0])

BATCH_SIZE = 256
BUFFER_SIZE = 10000

train_univariate = tf.data.Dataset.from_tensor_slices((x_train_uni, y_train_uni))
train_univariate = train_univariate.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

test_univariate = tf.data.Dataset.from_tensor_slices((x_test_uni, y_test_uni))
test_univariate = test_univariate.batch(BATCH_SIZE).repeat()

simple_lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(8, input_shape=x_train_uni.shape[-2:]),
    tf.keras.layers.Dense(1)
])

simple_lstm_model.compile(optimizer='adam', loss='mae')
EVALUATION_INTERVAL = 200 #epoch only runs for 200 steps as opposed to full data as is normal to save time
EPOCHS = 10
print(train_univariate)
# simple_lstm_model.fit(train_univariate, epochs=EPOCHS,
#                       steps_per_epoch=EVALUATION_INTERVAL,
#                       validation_data=test_univariate, validation_steps=50)

# mae = simple_lstm_model.evaluate(
#     test_univariate, steps = 200)

# print(f"test mae: {mae}")

'''
Now for part 2
'''