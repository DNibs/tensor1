# This application predicts output of continuous variable. In this case, it predicts
#   the median price of homes in a Boston suburb during mid 1970s.

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

import numpy as np

print(tf.__version__)

# Download and load Boston housing data from tensorflow
boston_housing = keras.datasets.boston_housing
(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

print("Training set: {}".format(train_data.shape)) #404 examples, 13 features
print("Testing set: {}".format(test_data.shape)) #102 examples, 13 features

# Features are as follows:
#   1. Per capita crime rate.
#   2. The proportion of residential land zoned for lots over 25,000 square feet.
#   3. The proportion of non-retail business acres per town.
#   4. Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
#   5. Nitric oxides concentration (parts per 10 million).
#   6. The average number of rooms per dwelling.
#   7. The proportion of owner-occupied units built before 1940.
#   8. Weighted distances to five Boston employment centers.
#   9. Index of accessibility to radial highways.
#   10. Full-value property-tax rate per $10,000.
#   11. Pupil-teacher ratio by town.
#   12. 1000 * (Bk - 0.63) ** 2 where Bk is the proportion of Black people by town.
#   13. Percentage lower status of the population.

# Note - scales differ per feature... therefore data will need to be "cleaned up"
print(train_data[0])


# pandas is a library for displaying data in tables and such
import pandas as pd

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
                'TAX', 'PTRATIO', 'B', 'LSTAT']
df = pd.DataFrame(train_data, columns=column_names)
df.head()

print(train_labels[0:10]) #display first 10 entries

# Normalize features
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

print(train_data[0]) #first training sample, normalized


# Create the Model
# Sequential model with two dense hidden layers, outputs single continuous value
# Model building steps wrapped in build_model() so that we can easily build second later

def build_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation=tf.nn.relu,
                           input_shape=(train_data.shape[1],)),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(1)
    ])

    optimizer = tf.train.RMSPropOptimizer(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae'])
    return model

model = build_model()
model.summary()


# Train the Model
# Trained for 500 epochs, record of training and validation in "history" object

# Display training progess for each completed epoch
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(selfself, epoch, logs):
        if epoch % 100 == 0: print(' ')
        print('.', end='')

EPOCHS = 500

# Store training stats
history  = model.fit(train_data, train_labels, epochs=EPOCHS,
                     validation_split=0.2, verbose=0,
                     callbacks=[PrintDot()])


# Visualize Training Progress
import matplotlib.pyplot as plt

def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [$1000]')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
             label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
             label = 'Val loss')
    plt.legend()
    plt.ylim([0,5])

plot_history(history)
#plt.show()

# Use callback to stop training when validation levels out
model = build_model()

# Patience parameter is amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

history = model.fit(train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[early_stop, PrintDot()])

plot_history(history)
plt.show()


# Measure Performance on test set
[loss, mae] = model.evaluate(test_data, test_labels, verbose=0)

print("\nTesting set Mean Abs Error: ${:7.2f}".format(mae * 1000))


# Predict housing prices using data in test set
test_predictions = model.predict(test_data).flatten()

print(test_predictions)