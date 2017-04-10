import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print('TensorFlow version: {0}'.format(tf.__version__))

hello = tf.constant('Hello, TensorFlow!')
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
print(sess.run(hello))
