import tensorflow as tf

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']



#preprocess the data

train_images = train_images / 255.0
test_images = test_images / 255.0

''' Uncomment the below if I wanna use the conv2d layer in MyModel() '''
# train_images  = train_images [..., tf.newaxis]
# test_images = test_images[..., tf.newaxis]


# train_ds = tf.data.Dataset.from_tensor_slices(
#     (train_images, train_labels)).shuffle(10000).batch(32)
# test_ds = train_ds = tf.data.Dataset.from_tensor_slices(
#     (test_images, test_labels)).batch(32)

train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(32).shuffle(10000)
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(32)
print(f"after preprocessing: {train_images.shape}")

class MyModel(tf.keras.Model):
  def __init__(self):
    super(MyModel, self).__init__()
    # self.conv1 = tf.keras.layers.Conv2D(32, 1, activation='relu')
    # self.flatten = tf.keras.layers.Flatten()
    # self.d1 = tf.keras.layers.Dense(128, activation='relu')
    # self.d2 = tf.keras.layers.Dense(10)
    self.flatten = tf.keras.layers.Flatten(input_shape = (28,28))
    self.d1 = tf.keras.layers.Dense(128, activation='relu')
    self.d2 = tf.keras.layers.Dense(10)

  def call(self, x):
    # using the functional api to do the feed-forward in this model subclassing
    # x = self.conv1(x)
    # x = self.flatten(x)
    # x = self.d1(x)
    # x = self.d2(x)
    x = self.flatten(x)
    x = self.d1(x)
    x = self.d2(x)
    return x




fashion_mnist_model = MyModel()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images,labels):
    with tf.GradientTape() as tape:
        predictions = fashion_mnist_model(images, training=True)
        print(f"first prediction: {predictions[0]}")
        error = loss_fn(labels,predictions)
    gradients = tape.gradient(error, fashion_mnist_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients,fashion_mnist_model.trainable_variables))
    
    train_loss(error)
    train_accuracy(labels, predictions) 


@tf.function
def test_step(images, labels):
  # training=False is only needed if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  predictions = fashion_mnist_model(images, training=False)
  t_loss = loss_fn(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)

EPOCHS = 15

for epoch in range(EPOCHS):
  # Reset the metrics at the start of the next epoch
  train_loss.reset_states()
  train_accuracy.reset_states()
  # test_loss.reset_states()
  # test_accuracy.reset_states()
  print("EPOCH is happening")
  for images, labels in train_ds:
    # print(images.shape)
    # print(labels.shape)
    train_step(images, labels)
    

  # personally don't think that iterating through test set every time makes sense
  # for test_images, test_labels in test_ds:
  #   test_step(test_images, test_labels)

  template = 'Epoch {}, Loss: {}, Accuracy: {}'
  print(template.format(epoch + 1,
                        train_loss.result(),
                        train_accuracy.result() * 100))

test_loss.reset_states()
test_accuracy.reset_states()
for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

print(f"Test loss: {test_loss.result()}, Test Accuracy: {test_accuracy.result()}")