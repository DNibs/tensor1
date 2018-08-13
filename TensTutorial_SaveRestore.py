# Intro to Savind and Restoring models throughout training process

from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf
from tensorflow import keras

print(tf.__version__)


# Download MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28*28) / 255.0
test_images = test_images[:1000].reshape(-1, 28*28) / 255.0


# Create simple sequential model
def create_model():
    model = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

    return model

# Create basic model instance
model = create_model()
model.summary()



# Create checkpoints to save model parameters during training
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create Checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

model = create_model()

model.fit(train_images, train_labels, epochs=10,
          validation_data = (test_images, test_labels),
          callbacks = [cp_callback]) #pass callback to training

# You should see some .ckpt files in the "tensor1/training_1" directory


# Create a new, untrained model. The accuracy is only chance, and therefore should be very low
model = create_model()

loss, acc = model.evaluate(test_images, test_labels)
print("Untrained model, accuracy: {:5.2f}%".format(100*acc))


# Now load the pre-trained weights from .ckpt files (made in line 54)
model.load_weights(checkpoint_path)
loss, acc = model.evaluate(test_images, test_labels)
print("Restored model 1, accuracy: {:5.2f}%".format(100*acc))



# Train a new model and save uniquely named checkpoint every 5 epochs
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    period=5 # Save weights every 5 epochs
)

model = create_model()
model.fit(train_images, train_labels,
          epochs=50, callbacks = [cp_callback],
          validation_data= (test_images, test_labels),
          verbose=0)


# Now look at resulting checkpoints sorted by modification date
import pathlib

checkpoints = pathlib.Path(checkpoint_dir).glob("*.index")
checkpoints = sorted(checkpoints, key=lambda cp:cp.stat().st_mtime)
checkpoints = [cp.with_suffix('') for cp in checkpoints]
latest = str(checkpoints[-1])
print(checkpoints)


# Reset model and load latest checkpoint
model = create_model()
model.load_weights(latest)
loss, acc = model.evaluate(test_images, test_labels)
print("Restored model 2, accuracy: {:5.2f}%".format(100*acc))



# Manually save weights
model.save_weights('./checkpoints/my_checkpoint')

# Restore the weights
model = create_model()
model.load_weights('./checkpoints/my_checkpoint')

loss, acc = model.evaluate(test_images, test_labels)
print("Restored model 3, accuracy: {:5.2f}%".format(100*acc))



# Save entire model
#   Saves weights, models configuration, and even optimizer configuration
#   Allows you to pause training and resume later from exact same state without access to original code
model = create_model()
model.fit(train_images, train_labels, epochs=5)

# Save entire model to HDF5 file
model.save('my_model.hdf5')

# Now recreate the entire model, including weights and optimizer
new_model = keras.models.load_model('my_model.hdf5')
new_model.summary()

# Check accuracy
loss, acc = new_model.evaluate(test_images, test_labels)
print("Restored new model, accuracy: {:5.2f}%".format(100*acc))

