import numpy as np
import tensorflow as tf

def predict_labels():
    pass

def get_MAE(true_labels, predicted_labels):
    """ Mean Absolute Error: 
    defined as average deviation of predicted class from true class """
    MAE = tf.reduce_mean(tf.abs(tf.diff(true_labels, predicted_labels)))
    return MAE

def get_MSE(true_labels, predicted_labels):
    """ Mean Squared Error """
    MSE = tf.reduce_mean(tf.square(tf.diff(true_labels, predicted_labels)))
    return MSE

def get_error_metric(true_labels, predicted_labels, metric="MAE"):
    errors = tf.diff(true_labels, predicted_labels)
    if metric=="MAE":
        metric = tf.abs(errors)
    elif metric=="MSE":
        metric = tf.square(errors)
    return tf.reduce_mean(metric)
