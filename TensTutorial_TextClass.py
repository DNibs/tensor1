# Binary Classification Example
# Classifies movie reviews as positive or negative using text of review
# https://www.tensorflow.org/tutorials/keras/basic_text_classification


import tensorflow as tf
from tensorflow import keras

import numpy as np

print(tf.__version__)

# Download dataset from IMDB (or use cached copy if it already exits)
imdb = keras.datasets.imdb

# num_words=10000 keeps 10000 most frequenstly used words and discards rare words for manageable set
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = 10000)

# Explore Data
# Data is preprocessed, 0 is negative and 1 is positive
print("Training entries: {}, labels{}".format(len(train_data), len(train_labels)))

# Text of reviews have been converted to integers, each representing a specific word
#print(train_data[0])

# Reviews are of different lenghts - since inputs to neural networks must be same lenght, this
#   will need to be resolved later
print( len(train_data[0]), len(train_data[1]) )


# Convert int back to text
# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# the first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2 #unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

# Now use decode_review to display text for first review
print( decode_review(train_data[0]) )


# Prepare the Data
# This can be done in two ways: One-hot-encode the arrays to convert them into vectors of 0s and 1s.
#   For example, the sequence [3, 5] would become a 10,000-dimensional vector that is all zeros except
#   for indices 3 and 5, which are ones. Then, make this the first layer in our network—a Dense layer—that
#   can handle floating point vector data. This approach is memory intensive, though, requiring a
#   num_words * num_reviews size matrix
# Alternatively, we can pad the arrays so they all have the same length, then create an integer tensor of
#   shape num_examples * max_length. We can use an embedding layer capable of handling this shape as the
#   first layer in our network. We will use this second approach.

# To ensure same length, we use pad sequences to standardize
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

# New length of examples
print( len(train_data[0]), len(train_data[1]))
print(train_data[0])


# Build the Network
# Embedding layer: This layer takes the integer-encoded vocabulary and looks up the embedding vector
#   for each word-index. These vectors are learned as the model trains. The vectors add a dimension to
#   the output array. The resulting dimensions are: (batch, sequence, embedding)
# GlobalAveragePooling1D: returns a fixed-length output vector for each example by averaging over the sequence
#   dimension. This allows the model can handle input of variable length, in the simplest way possible.
# Fixed Length Output Vector is piped through dense layer with 16 hidden units.
# Final Layer: densely connected with a single output node. Using the sigmoid activation function, this value is
#   a float between 0 and 1, representing a probability, or confidence level.

# input shape is vocab count used for reviews - 10000 words
vocab_size = 10000

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D() )
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()


# Compile with loss function and optimizer
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Create validation set - set apart 10000 examples from original training data
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]


# Train the Model - 20 epochs in mini-batches of 512 samples. This is 20 iterations over all samples in
#   x_train and y_train tensors
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)


#Evaluate the Model - prints Loss and Accuracy
# NOTE- my 'results' don't seem to be accurate, despite graphs later showing accurate results
#   not sure what causes discrepancy between example and my app (or between printed results and graphs)
results = model.evaluate(test_data, test_labels)
print(results)


# Create graph of accuracy and loss over time
history_dict = history.history
history_dict.keys()

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# bo is for blue dot
plt.figure()
plt.plot(epochs, loss, 'bo', label = 'Training loss')
# b is for solid blue line
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.draw()


#plt.clf() # clear figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.figure()
plt.plot(epochs, acc, 'bo', label = 'Training acc')
plt.plot(epochs, val_acc, 'b', label = 'Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.draw()

plt.show()

