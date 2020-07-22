import tensorflow as tf
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
url = "https://raw.githubusercontent.com/bharddwaj/Summer2020/master/US1.IBM_190716_200715.csv"
stock = pd.read_csv(url)

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
start_index = 0
end_index = 87000
end_test_index = 88000
history_size = 10
target_size = 1
x_train, y_train = univariate_data(start_index,end_index, history_size,target_size,ibm_returns)
x_test, y_test = univariate_data(end_index,end_test_index,history_size,target_size,ibm_returns)

train_univariate = tf.data.Dataset.from_tensor_slices(x_train)
train_univariate = train_univariate.cache().batch(BATCH_SIZE)

test_univariate = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_univariate = test_univariate.batch(BATCH_SIZE).repeat()
tf.keras.backend.set_floatx('float32') #to change all layers to have this dtype by default

def make_generator():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(5*2, use_bias=False, input_shape=(100,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU())
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(10))
    model.add(tf.keras.layers.Reshape((10, 1)))
    return model


def make_discriminator():
    '''
    why i had to make the default activation function different
    https://github.com/tensorflow/tensorflow/issues/30263
    '''
    model = tf.keras.Sequential([tf.keras.layers.LSTM(units = 8, activation = 'sigmoid', input_shape = x_train.shape[-2:]), \
    tf.keras.layers.Dense(units = 1)])
    return model


generator = make_generator()
discriminator = make_discriminator()

noise = tf.random.normal([1, 100])
generated_image = generator(noise)
print(generated_image.shape)

# '''
# https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial
# '''
# import os

# os.environ['KMP_DUPLICATE_LIB_OK']='True'
# '''
# this os stuff is to fix he libiomp5.dylib error
# '''
# plt.imshow(generated_image[0, :, :, 0], cmap='gray')
# plt.show()

discriminator = make_discriminator()
decision = discriminator(generated_image)
print (decision)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

EPOCHS = 50
noise_dim = 100


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)
      print("generated image")
      real_output = discriminator(images, training=True)
      print("real output")
      fake_output = discriminator(generated_images, training=True)
      print("fake output")

      gen_loss = generator_loss(fake_output)
      print(f"Gen Loss: {gen_loss}")
      disc_loss = discriminator_loss(real_output, fake_output)
      print(f"disc_loss: {disc_loss}")

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for batch in dataset:
      train_step(batch)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))


train(train_univariate, EPOCHS)

gen_data = generator(noise)
decision = discriminator(gen_data)
print(decision) 


