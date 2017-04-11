import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print('TensorFlow version: {0}'.format(tf.__version__))

hello = tf.constant('Hello, TensorFlow!')
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
# also tf.float32 implicitly
print(node1, node2)


sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# print(sess.run(hello))

# sess = tf.Session()
print(sess.run([node1, node2]))

node3 = tf.add(node1, node2)
print("node3: ", node3)
print("sess.run(node3): ", sess.run(node3))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b

print(sess.run(adder_node, {a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))