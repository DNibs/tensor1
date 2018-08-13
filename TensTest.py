import tensorflow as tf

#ignores warning for CPU not using AVX2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#define constants 'a' and 'b'
a = tf.constant(1.0, dtype = tf.float32, name = "a")
b = tf.constant(2.0, dtype = tf.float32, name = "b")

#compute sum 'a+b"
s = tf.add(a, b, name="sum")
print(s)

#intialize session
with tf.Session() as sess:

    #execute graph
    result = sess.run(s)
    print(result)



#produce graph of multivariate distribution
import matplotlib.pyplot as plt
import numpy as np

mean = [0, 0]
cov = [[1, 0], [0, 100]]

x, y = np.random.multivariate_normal(mean, cov, 5000).T
plt.plot(x, y, 'x')
plt.axis('equal')
plt.show()