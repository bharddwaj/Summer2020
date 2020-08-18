import tensorflow as tf
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns


def generate_brownian():
  print("generating brownian motion")
  random_numbers = tf.random.normal((880000, 1), 0, (0.2/(365*24*60)**0.5))
  x = []
  summation = 0
  for i in range(len(random_numbers)):
    print(i)
    # x.append(sum(random_numbers[:i]))
    # assert sum(random_numbers[:i]) == summation
    x.append(summation)
    summation += random_numbers[i]
  x.pop(0)
  log_list = []
  for i in range(len(x[1:])):
    print(i)
    log_list.append(tf.math.log(abs(x[i]/x[i-1])))
  print("Finished generating brownian motion")
  return random_numbers, x, log_list
log_numbers, x, z = generate_brownian()

list_log_numbers = []
i = 0
while i <= log_numbers.shape[0]:
  print(i)
  a = []
  for j in range(10):
    a.append(log_numbers[i+j][0].numpy())
  try:
    i += 10 #to avoid repeats
  except:
    pass
  print(i)
  list_log_numbers.append(np.asarray(a).reshape((10,1)))

print("Finished the list_log_numbers")
BATCH_SIZE = 8800
BUFFER_SIZE = 10000


train_univariate = tf.data.Dataset.from_tensor_slices(x_train)
train_univariate = train_univariate.cache().batch(BATCH_SIZE)

test_univariate = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_univariate = test_univariate.batch(BATCH_SIZE).repeat()
tf.keras.backend.set_floatx('float32') #to change all layers to have this dtype by default

def make_generator():
    # model = tf.keras.Sequential()
    # model.add(tf.keras.layers.Dense(5*2, use_bias=False, input_shape=(100,)))
    # model.add(tf.keras.layers.BatchNormalization())
    # model.add(tf.keras.layers.LeakyReLU())
    # model.add(tf.keras.layers.Dense(32, activation='relu'))
    # model.add(tf.keras.layers.Dense(10))
    # model.add(tf.keras.layers.Reshape((10, 1)))
    model = tf.keras.Sequential(
        [tf.keras.layers.LSTM(units = 8, name="lstm1" ,activation = 'tanh', input_shape = (10,1), return_sequences = True),
        #  tf.keras.layers.LSTM(units = 32, activation = 'tanh', return_sequences = True),
        #  tf.keras.layers.LSTM(units = 32, activation = 'tanh', return_sequences = True),
         tf.keras.layers.LSTM(units = 32, activation = 'tanh'),
         tf.keras.layers.Dense(units = 10)])
    model.add(tf.keras.layers.Reshape((10, 1)))
    return model


def make_discriminator():
    '''
    why i had to make the default activation function different
    https://github.com/tensorflow/tensorflow/issues/30263
    '''
    model = tf.keras.Sequential(
        [tf.keras.layers.LSTM(units = 8, name="disc1", activation = 'tanh', input_shape = (10,1),return_sequences = True),
        #  tf.keras.layers.LSTM(units = 32, name="disc2", activation = 'tanh',return_sequences = True),
        #  tf.keras.layers.LSTM(units = 32, name="disc3", activation = 'tanh',return_sequences = True),
         tf.keras.layers.LSTM(units = 32, name="disc4", activation = 'tanh'),
         tf.keras.layers.Dense(units = 10, name="disc5")])
    return model



generator = make_generator()
print(generator.summary())
discriminator = make_discriminator()

noise = tf.random.normal(shape =(10,1))
noise = np.array([noise])
generated_image = generator(noise)
print(f"generated image: {generated_image.shape}")

'''
https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial
'''
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
'''
# this os stuff is to fix he libiomp5.dylib error
# '''
# plt.imshow(generated_image[0, :, :, 0], cmap='gray')
# plt.show()

discriminator = make_discriminator()
decision = discriminator(generated_image)
print (decision)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    # real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    # fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    # total_loss = real_loss + fake_loss
    # return total_loss
    return tf.math.reduce_mean(tf.math.negative(real_output - fake_output)) # min -loss same as max of loss

def generator_loss(fake_output):
    '''in the future: look for a better loss fn for the generator'''
    # all_distances = tf.math.reduce_mean(fake_output) #using this to center the returns around 0
    # expected_center = 0
    # alpha = 3
    # return cross_entropy(tf.ones_like(fake_output), fake_output) + alpha*(all_distances - expected_center)**2
    return tf.math.reduce_mean(tf.math.negative(fake_output))

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

EPOCHS = 10
noise_dim = 10

gradients_generator = []
gradients_discriminator = []

losses_generator = []
losses_discriminator = []


# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    # noise = tf.random.normal([BATCH_SIZE, noise_dim,1])
    noise = tf.random.uniform(shape = (10,1))
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)
      # print("generated image")
      real_output = discriminator(images, training=True)
      # print("real output")
      fake_output = discriminator(generated_images, training=True)
      # print("fake output")

      gen_loss = generator_loss(fake_output)
      # print(f"Gen Loss: {gen_loss}")
      disc_loss = discriminator_loss(real_output, fake_output)
      # print(f"disc_loss: {disc_loss}")

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    # print(f"Gen Gradient: {gradients_generator}")
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    # print(f"Discriminator Gradient: {gradients_of_discriminator}")

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    #testing to see whats wrong with the sigmoid
    gradients_generator.append(gradients_of_generator)
    gradients_discriminator.append(gradients_of_discriminator)

    losses_generator.append(gen_loss)
    losses_discriminator.append(disc_loss)

def train(dataset, epochs):
  for epoch in range(epochs):
    start = time.time()

    for batch in dataset:
      train_step(batch)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

print("About to start the training")
train(train_univariate, EPOCHS)
# generator = tf.keras.models.load_model('saved_models/generator')
# discriminator = tf.keras.models.load_model('saved_models/discriminator')
# gen_data = generator(noise)
# decision = discriminator(gen_data)
# print(gen_data.numpy()[0]) 
# plt.plot(list(range(gen_data.shape[1])), gen_data.numpy()[0])
# plt.show()
# generator.save('saved_models/generator')
# discriminator.save('saved_models/discriminator')

# compare the statistics of the generated data to the real data
# make a histogram of the generated first minutes and see if it creates a normal distribution
# can make the generator loss also include functions that make sure that the statisical properties are retained. cross entropy + normality (example)

## Let us see if the first data point of generated data is similar to normal distribution
'''
first_points = []
for i in range(10000):
  noise = tf.random.uniform([BATCH_SIZE, noise_dim])
  # noise = tf.random.normal([BATCH_SIZE, noise_dim,1])
  gen_data = generator(noise)
  # print(gen_data.numpy()[0][0][0]) 
  first_points.append(gen_data.numpy()[0][0][0])
  print(i)
  plt.plot(list(range(gen_data.shape[1])), gen_data.numpy()[0])
plt.show()
first_points = np.asarray(first_points)
# ax = sns.distplot(first_points)
plt.hist(first_points, bins = 30, edgecolor='black')
'''

'''
displays the training data
'''
# for i in range(x_train.shape[0]):
#   print(i)
#   plt.plot(list(range(10)),x_train[i])
# plt.show()



# losses_discriminator = list(map(lambda x: x.numpy(),losses_discriminator))
# print(losses_generator)
# print(losses_discriminator)
# plt.plot(range(len(losses_generator)), losses_generator)
# plt.show()
# plt.plot(range(len(losses_discriminator)), losses_discriminator)
# plt.show()

# print(len(gradients_generator[0]))
