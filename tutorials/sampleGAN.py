import tensorflow as tf
import numpy as np
import math
import random

x = [float(i) for i in range(1,100000)]
y = list(map(math.tan,x))
print(y)
x = tf.Variable(x)
y = tf.Variable(y)
data = zip(x,y)
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape = [1]))
model.add(tf.keras.layers.Dense(32, activation = "relu"))
model.add(tf.keras.layers.Dense(32, activation = "relu"))
model.add(tf.keras.layers.Dense(1))

loss_fn = tf.keras.losses.MSE
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])
model.fit(x,y, epochs  = 40)
print(f"log: {math.tan(157000), math.tan(123908)}")
print(f"model prediction: {model.predict([157000,123908])}")
