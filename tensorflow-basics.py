import tensorflow as tf
print(tf.__version__)

string = tf.Variable("this is a string", tf.string)
number = tf.Variable(324, tf.int16)
floating = tf.Variable(3.567, tf.float64)


rank1_tensor = tf.Variable(['Test'], tf.string)
rank2_tensor = tf.Variable([['test','ok','ok'],['test','yes','ok'],['ok','ok','ok']], tf.string)

tensor1 = tf.ones([1,2,3])
tensor2 = tf.reshape(tensor1, [2,3,1])
tensor3 = tf.reshape(tensor1,[3, -1])

tf.print(tf.add(number,number))