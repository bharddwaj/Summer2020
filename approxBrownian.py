import tensorflow as tf
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import scipy
import math

def generate_brownian():
  print("generating brownian motion")
  random_numbers = tf.random.normal((88001, 1), 0, (1/(365*24*60)**0.5))
  b = []
  summation = 0
  starting_stock_price = 10000
  s = [starting_stock_price]
  log_s = []
  for i in range(len(random_numbers)):
    print(i)
    # x.append(sum(random_numbers[:i]))
    # assert sum(random_numbers[:i]) == summation
    b.append(summation)
    if i > 0:
      s.append(s[i-1]*tf.math.exp(summation))
      log_s.append(math.log(s[i])-math.log(s[i-1]))
    summation += random_numbers[i]
  
  print("Finished generating brownian motion")
  return b,s,log_s
b,s,log_s = generate_brownian()
print(len(log_s))
i = 0
j = 0
counter = 0
list_log_s = np.zeros(shape = (880,100))
while i < len(log_s):
  list_log_s[counter,j] = log_s[i]
  i+=1
  if j == 99:
    counter +=1
    j = 0
    print(counter)
  else:
    j+=1
print(len(list_log_s))
# log_s = np.asfarray(list(map(lambda x: x.numpy()[0], log_s)))
print(log_s)
print(np.mean(log_s))
print(np.std(log_s))
print(scipy.stats.skew(log_s))
print(scipy.stats.kurtosis(log_s))

EPOCHS = 20
BATCH_SIZE = 880 
BUFFER_SIZE = 10000
start_index = 0
end_index = 87000
end_test_index = 88000
history_size = 10
target_size = 1
noise_dim = 100
list_log_s2 = np.asfarray(list(map(lambda x: x.reshape((100,1)),list_log_s)))
print(len(list_log_s2))
train_univariate = tf.data.Dataset.from_tensor_slices(list_log_s2)
train_univariate = train_univariate.cache().batch(BATCH_SIZE)


tf.keras.backend.set_floatx('float64') #to change all layers to have this dtype by default

def make_generator():
    model = tf.keras.Sequential(
        [tf.keras.layers.LSTM(units = 8, name="lstm1" ,activation = 'tanh', input_shape = (noise_dim,1), return_sequences = True),
         tf.keras.layers.LSTM(units = 32, activation = 'tanh'),
         tf.keras.layers.Dense(units = noise_dim)])
    model.add(tf.keras.layers.Reshape((noise_dim, 1)))
    return model


def make_discriminator():
    model = tf.keras.Sequential(
        [tf.keras.layers.LSTM(units = 8, name="disc1", activation = 'tanh', input_shape = (noise_dim,1),return_sequences = True),
         tf.keras.layers.LSTM(units = 32, name="disc4", activation = 'tanh'),
         tf.keras.layers.Dense(units = noise_dim, name="disc5")])
    return model



generator = make_generator()
discriminator = make_discriminator()

noise = tf.random.normal(shape =(noise_dim,1))
noise = np.array([noise])

generated_image = generator(noise)
print(f"generated image: {generated_image.shape}")

discriminator = make_discriminator()
decision = discriminator(generated_image)
print(decision)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    return tf.math.reduce_mean(tf.math.negative(real_output - fake_output)) # min -loss same as max of loss
def generator_loss(fake_output):
    return tf.math.reduce_mean(tf.math.negative(fake_output))
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


gradients_generator = []
gradients_discriminator = []

losses_generator = []
losses_discriminator = []


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    # noise = tf.random.normal([BATCH_SIZE, noise_dim,1])
    noise = tf.random.uniform([BATCH_SIZE, noise_dim, 1])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)
      print("generated image")
      real_output = discriminator(images, training=True)
      print("real output")
      fake_output = discriminator(generated_images, training=True)
      print("fake output")

      gen_loss = generator_loss(fake_output)
      print(f"Gen Loss: {gen_loss.shape}")
      disc_loss = discriminator_loss(real_output, fake_output)
      print(f"disc_loss: {disc_loss.shape}")

      gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
      # print(f"Gen Gradient: {gradients_generator}")
      gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
      # print(f"Discriminator Gradient: {gradients_of_discriminator}")

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss,disc_loss

def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for batch in dataset:
      gen_loss,disc_loss = train_step(batch)
      losses_generator.append(gen_loss)
      losses_discriminator.append(disc_loss)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

print("About to start the training")
print(train_univariate)
train(train_univariate, EPOCHS)
print("done training")
print(f"Gen Loss: {losses_generator}")
print(f"Disc Loss: {losses_discriminator}")
points = []
for i in range(100):
  noise = tf.random.uniform([BATCH_SIZE, noise_dim,1])
  # noise = tf.random.normal([BATCH_SIZE, noise_dim,1])
  gen_data = generator(noise)
  print(gen_data.numpy()) 
  points.append(gen_data.numpy())
  print(i)
  break
points = np.asarray(points)
