import numpy as np
import tensorflow as tf

def predict_labels(class_probs):
    predicted_labels = tf.argmax(class_probs, axis=1)
    return predicted_labels

def get_MAE(true_labels, predicted_labels):
    """ Mean Absolute Error: 
    defined as average deviation of predicted class from true class """
    MAE = tf.reduce_mean(tf.abs(tf.subtract(true_labels, predicted_labels)))
    return MAE

def get_MSE(true_labels, predicted_labels):
    """ Mean Squared Error """
    MSE = tf.reduce_mean(tf.square(tf.subtract(true_labels, predicted_labels)))
    return MSE

def get_error_metric(true_labels, predicted_labels, metric="MAE"):
    errors = tf.subtract(true_labels, predicted_labels)
    if metric=="MAE":
        metric = tf.abs(errors)
    elif metric=="MSE":
        metric = tf.square(errors)
    return tf.reduce_mean(metric)
