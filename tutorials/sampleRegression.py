# tutorial
# https://www.tensorflow.org/tutorials/keras/regression
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
'''
https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial
'''
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
'''
this os stuff is to fix he libiomp5.dylib error
'''
dataset_path = tf.keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
print(dataset.tail())

# dataset.isna().sum()

dataset = dataset.dropna()

# convert this categorical variable to one hot vector
dataset['Origin'] = dataset['Origin'].map({1: 'USA', 2: 'Europe', 3: 'Japan'})

dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')
print(dataset.tail())
train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
# plt.show()

train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()

# MPG is what we are trying to predict using the model
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

def normalize(x):
    return (x - train_stats['mean']) / train_stats['std']

normed_train_data = normalize(train_dataset)
normed_test_data = normalize(test_dataset)

def build_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model


EPOCHS = 1000

model = build_model()

# The patience parameter is the amount of epochs to check for improvement
# early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

early_history = model.fit(normed_train_data.to_numpy(), train_labels.to_numpy(), 
                    epochs=EPOCHS, validation_split = 0.2, verbose=0)

loss, mae, mse = model.evaluate(normed_test_data.to_numpy(), test_labels.to_numpy(), verbose=2)

print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))