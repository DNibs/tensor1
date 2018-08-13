# This is a tutorial neural netowrk that classifies imgs from hte fashion.mnist dataset
# It uses only input layer and softmax layer
# Found on https://www.tensorflow.org/tutorials/keras/basic_classification


#TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

#Helper Libraries
import numpy as np
import matplotlib.pyplot as plt



print(tf.__version__)

#Download and load the "fashion" mnist dataset (if already downloaded, it simply loads to app)
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#Verify data - (60000, 28, 28)   60000    [9 0 0 ... 3 0 5]   10000
print(train_images.shape)
print(len(train_labels))
print(train_labels)
print(len(test_labels))

#Define the classes (which are currently just labeled 0-9
class_names = ['T-shirt/top', 'Trouser', 'Pullover/CARDIGAN', 'Dress', 'Coat', 'Sandal', 'Shirt',
               'Sneaker', 'Bag', 'Ankle boot']

#Show first image (also verifies data working....)
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.gca().grid(False)
#plt.show(0)
plt.draw()

#Scale data from (0-255) to (0-1), otherwise classifier might be skewed
train_images = train_images / 255.0
test_images = test_images / 255.0


#Display first 25 imgs from training set and class name below each image (more verification)
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])

#plt.show(1)
#NOTE - might have to close graphs for code to continue
plt.draw()

#Begin building the network!
#Set up layers
#First layer transforms imgs from 2D array to 1D array
#Second Layer fully connected, 128 node
#Final layer fully connected, 10 node (for each class), softmax that returns probability for each class
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

#compile model by defining loss function, optimizer, and metrics
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Call model.fit to start training. Will "feed" data, associate img/labels, make predictions
model.fit(train_images, train_labels, epochs=5)

#Evaluate accurace - see how model performs on test set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

#With a trained a verified model, make predictions on some images
predictions = model.predict(test_images)

predictions[0] #gives array of 10 numbers for img 0, corresponding to "confidence" of each class
np.argmax(predictions[0]) #ID class with highest confidence
test_labels[0] #The "ground truth" label for img 0

# Plot the first 25 test images, their predicted label, and the true label
# Color correct predictions in green, incorrect in red
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions[i])
    true_label = test_labels[i]
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'
    plt.xlabel("{} ({})".format(class_names[predicted_label], class_names[true_label]), color = color)
#plt.show(2)
plt.draw()

# Use trained model to make prediction about single image
img = test_images[0]
print(img.shape)

# Make batch from single image (since keras models make predictions on batches)
img = (np.expand_dims(img, 0))
print(img.shape)

# Now make prediction
predictions = model.predict(img)
print(predictions)

# Grab the prediction from the results (array of batch, even if batch is only 1 element)
prediction = predictions[0]
np.argmax(prediction)
print( np.argmax(prediction)) #should show '9'

plt.show()