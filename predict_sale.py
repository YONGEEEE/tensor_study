import tensorflow as tf
import matplotlib.pyplot as plt

xData = [1, 2, 3, 4, 5, 6, 7]
yData = [25000,55000, 75000, 110000, 128000, 155000, 180000]
W = tf.Variable(tf.random_uniform([1], -100, 100))
b = tf.Variable(tf.random_uniform([1], -100, 100))
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

H = W * X + b #Hypothesis
cost = tf.reduce_mean(tf.square(H - Y)) #Cost function

a = tf.Variable(0.01)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)# cost 최소화(gradent descent algorithm)
# learning_rate = 0.1
# gradient = tf.reduce_mean((W*X-Y)*X)
# descent = W-learning_rate*gradient
# update = W.assign(descent)

init = tf.global_variables_initializer()# Variable 변수 초기화

sess = tf.Session()
sess.run(init)

for i in range(5001):
    cost_val, hy_val, _ =sess.run([cost, H, train], feed_dict= {X: xData, Y : yData})
    if i % 500 == 0:
        print (i,'Cost:',cost_val,'prediction:',hy_val)
print (sess.run(H, feed_dict={X : [8]}))



