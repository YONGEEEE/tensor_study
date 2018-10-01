import tensorflow as tf
hello = tf.constant('Hello, TensorFlow')#tensor
sess = tf.Session()
print(sess.run(hello))