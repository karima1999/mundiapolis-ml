#!/usr/bin/env python3

import tensorflow as tf


def calculate_accuracy(y, y_pred):
	
    accuracy = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    mean = tf.reduce_mean(tf.cast(accuracy, tf.float32))
    return mean