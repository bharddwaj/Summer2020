import tensorflow as tf
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import scipy
import math
'''
https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial
'''
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
'''
this os stuff is to fix he libiomp5.dylib error
'''

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
      #s.append(s[i-1]*math.exp(summation))
      s.append(s[i-1]*math.exp(random_numbers[i]))
      log_s.append(math.log(s[i])-math.log(s[i-1]))
    summation += random_numbers[i]
  
  print("Finished generating brownian motion")
  return b,s,log_s
b,s,log_s = generate_brownian()

def split_into_batches(data,x,y):
  if x*y != len(data):
    raise ValueError(f"{x*y} does not equal {len(data)}")
  else:
    i = 0
    j = 0
    counter = 0
    data_for_network = np.zeros(shape = (x,y))
    while i < len(data):
      data_for_network[counter,j] = data[i]
      i+=1
      if j == y-1:
        counter +=1
        j = 0
        print(counter)
      else:
        j+=1
  
    return data_for_network.reshape(x,y,1)  #np.asfarray(list(map(lambda x: x.reshape((y,1)),data_for_network)))

list_log_s = split_into_batches(log_s,880,100)
print(f"list log: {list_log_s.shape}")

EPOCHS = 100
BATCH_SIZE = 880 
noise_dim = 100

train_univariate = tf.data.Dataset.from_tensor_slices(list_log_s)
train_univariate = train_univariate.cache().batch(BATCH_SIZE)
tf.keras.backend.set_floatx('float64') #to change all layers to have this dtype by default

def make_generator():
    model = tf.keras.Sequential([tf.keras.Input(shape=(noise_dim,1), name = 'input_gen' ),
                                 tf.keras.layers.Dense(32, activation = 'relu', name = '1st_gen'),
                                 tf.keras.layers.Dense(32, activation = 'relu', name = '2nd_gen'),
                                 tf.keras.layers.Dense(1, name = 'output_gen')])
    return model

def make_discriminator():
    model = tf.keras.Sequential([tf.keras.Input(shape=(noise_dim,1),name = 'input_dis'),
                              tf.keras.layers.Dense(32, activation = 'relu',name = '1st_dis'),
                              tf.keras.layers.Dense(32, activation = 'relu', name = '2nd_dis'),
                              tf.keras.layers.Dense(1, name = 'output_dis', activation = 'sigmoid')])
    return model

generator = make_generator()
discriminator = make_discriminator()
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    return cross_entropy(tf.ones_like(real_output),real_output) + cross_entropy(tf.zeros_like(fake_output),fake_output)
    
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output),fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

gradients_generator = []
gradients_discriminator = []

losses_generator = []
losses_discriminator = []

@tf.function
def train_step(images):
    '''only train discriminator'''
    noise = tf.random.uniform([BATCH_SIZE, noise_dim, 1])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)
      
      real_output = discriminator(images, training=True)
      
      fake_output = discriminator(generated_images, training=True)
      
      gen_loss = generator_loss(fake_output)
      
      disc_loss = discriminator_loss(real_output, fake_output)
      

      gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
      
      gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
  
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss,disc_loss,gradients_of_generator, gradients_of_discriminator

def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for batch in dataset:
      
      gen_loss,disc_loss, grads_generator, grads_discriminator = train_step(tf.reshape(batch,(BATCH_SIZE,noise_dim,1)))
      losses_generator.append(gen_loss)
      losses_discriminator.append(disc_loss)
      gradients_generator.append(grads_generator)
      gradients_discriminator.append(grads_discriminator)
    
  
    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

print("About to start the training")
train(train_univariate, EPOCHS)
print("done training")

generations = generator(tf.random.uniform([BATCH_SIZE, noise_dim,1])).numpy()

this_generation = generations.flatten()
plt.hist(log_s, bins = 200, color="red")
plt.show()
plt.hist(this_generation, bins = 200, color="black")
plt.show()

print(f"Actual Normal: {scipy.stats.jarque_bera(log_s)}")
print(f"GAN output: {scipy.stats.jarque_bera(this_generation)}")
