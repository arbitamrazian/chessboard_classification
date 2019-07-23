import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=np.nan)

def get_board_preds(prediction):
    xs = tf.split(prediction,64,axis=1)
    output = []
    for x in xs:
        casted = tf.where(
            tf.equal(tf.reduce_max(x, axis=1, keep_dims=True), x)
        )
        output.append(casted)
    return tf.concat(output,axis=1)

x_np = np.random.beta(1,1,13*64)
y_np = np.random.beta(1,1,13*64)
a_np = np.random.beta(1,1,13*64)
c_np = np.random.beta(1,1,13*64)
b_np = np.random.beta(1,1,13*64)

x = tf.Variable(x_np)
y = tf.Variable(y_np)
a = tf.Variable(b_np)
b = tf.Variable(a_np)
c = tf.Variable(c_np)

data = tf.stack([x,y,c,b,a])


with tf.Session() as sess:
    labels = tf.constant([[1, 0, 0, 1],
			  [0, 1, 1, 1],
			  [1, 1, 0, 0],
			  [0, 0, 0, 1],
			  [1, 1, 0, 0]])

    preds = tf.constant([[1, 1, 1, 1],
			 [1, 1, 1, 1],
			 [1, 1, 1, 1],
			 [1, 1, 1, 1],
			 [1, 1, 1, 1]])

    acc, acc_op = tf.metrics.accuracy(labels=labels, predictions=preds) 
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    print(sess.run([acc, acc_op]))
    print(sess.run([acc]))
