
import tensorflow as tf

def create_placeholders(nx, classes):
    
    x = tf.placeholder(tf.float32, shape=(None, nx))
    y = tf.placeholder(tf.float32, shape=(None, classes))
    return x, y
