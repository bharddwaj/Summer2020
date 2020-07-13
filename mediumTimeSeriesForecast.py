# https://towardsdatascience.com/time-series-forecasting-with-lstms-using-tensorflow-2-and-keras-in-python-6ceee9c6c651
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc



sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 16, 10

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

time = np.arange(0, 100, 0.1)
data = np.sin(time) + np.random.normal(scale=0.5, size=len(time))
# print(data)
# print("about to plot the data")
# plt.plot(time,data)
# plt.show()
time = np.array(time, dtype = 'float32')

# trying to mimic the above with the tensorflow library
data2 = tf.math.sin(time) + tf.random.normal([len(time)],stddev = 0.5,dtype = tf.float32)

# data = list(map(lambda x: round(x,4), data))
# data2 = list(map(lambda x: round(x,4), data2.numpy()))

# for i in range(len(data)):
#     if data[i] != data2[i]:
#         print("Not Equal")
#         print(data[i])
#         print(data2[i])
#         break

df = pd.DataFrame(dict(actual_data=data), index=time, columns=['actual_data'])
df2 = pd.DataFrame(dict(actual_data2=data2), index=time, columns=['actual_data2'])

train_size = int(len(df) * 0.8)
test_size = len(df) - train_size
train, test = df.iloc[0:train_size], df.iloc[train_size:len(df)]
train2, test2 = df2.iloc[0:train_size], df2.iloc[train_size:len(df2)]
print(f"Dataframe: {df}")
print(len(train), len(test))

# We need to predict the valuea t the current time step by using the history (n time steps from it)
def create_dataset(X,y,time_steps = 1):
    '''
    Works with univariate and multivariate time series data
    '''
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 10

# reshape to [samples, time_steps, n_features]

X_train, y_train = create_dataset(train, train.actual_data, time_steps)
X_test, y_test = create_dataset(test, test.actual_data, time_steps)

X_train2, y_train2 = create_dataset(train2, train2.actual_data2, time_steps)
X_test2, y_test2 = create_dataset(test2, test2.actual_data2, time_steps)

print(X_train.shape, y_train.shape)

# shape: (samples, time_steps, features)

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(
  units=128,
  input_shape=(X_train.shape[1],X_train.shape[2])
))
model.add(tf.keras.layers.Dense(units=1))
model.compile(
  loss='mean_squared_error',
  optimizer= tf.keras.optimizers.Adam(0.001),
  metrics = ["accuracy"]
)
# do NOT shuffle the data for Time Series becasuse the order matters

history = model.fit(X_train,y_train,epochs = 30, validation_split = 0.1,shuffle = False, verbose = 1)
'''
https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial
'''
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
'''
this os stuff is to fix he libiomp5.dylib error
'''
print("About to print the metric names")
print(model.metrics_names)
print("done")

plt.plot(history.history['loss'], color = "blue")
plt.plot(history.history['val_loss'], color = "red")
# plt.plot(history.history['accuracy']) jus gives a horizontal line 0
plt.show()