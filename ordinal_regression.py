import numpy as np
import tensorflow as tf
import pandas as pd

from ordinal_eval import predict_labels, get_error_metric
epsilon = 1e-30

def load_sample_data(filename):
    """ Note, can also get the boston dataset from sklearn """
    data = pd.read_csv(filename, header=None)
    X = data[[0, 1, 2, 3]].values
    y = data[4].values
    return X, y

def get_ordinal_weights(num_dims):
    weights = tf.get_variable("ordinal_weights", shape = [num_dims, 1])
    return weights

def get_ordinal_thresholds(num_thresholds, init_vals = None):
    """ Providing threshold_range implies defining own thresholds, not learning them """
    if init_vals is not None:
        ordinal_thresholds = tf.constant(init_vals)
    else:
        shape = [num_thresholds, 1]
        unordered_thresholds = tf.Variable(tf.random_uniform(shape, minval=-threshold_scale,
                                                             maxval = threshold_scale),
                                                             name = "ord_thresholds")
        ordinal_thresholds = tf.contrib.framework.sort(unordered_thresholds)
    return ordinal_thresholds

def compute_ordinal_vars(weights, input_features):
    input_features = tf.cast(input_features, tf.float32)
    ordinal_vars = tf.matmul(input_features, weights, transpose_a = False, transpose_b=False)
    return ordinal_vars

def threshold_var_diff(ordinal_vars, thresholds):
    """ The difference is accomplished via broadcasting.
    Does c_l - z_it for all thresholds c_l """
    thresholds = tf.cast(thresholds, tf.float32)
    ordinal_vars = tf.cast(ordinal_vars, tf.float32)
    differences = tf.subtract(thresholds, ordinal_vars)
    return differences

def prob_all_classes(logistic_matrix, num_classes):
    """ Do column differences of logistic matrix
    Note that there is an implicit appending of 1's for last column and 0's for first """
    probs_list = []
    for ord_class in range(num_classes):
        prob_class = prob_of_class(ord_class, logistic_matrix, num_classes)
        probs_list.append(prob_class)
    probs_all_classes = tf.stack(probs_list, axis=1)
    return probs_all_classes

def prob_of_class(class_idx, logistic_matrix, num_classes):
    class_column = tf.gather(logistic_matrix, class_idx, axis=1)
    class_column = tf.cast(class_column, tf.float32)
    if class_idx == num_classes-1:
        neighbor_column = 1.0
        class_diff = tf.subtract(neighbor_column, class_column)
    elif class_idx == 0:
        neighbor_column = 0.0
        class_diff = tf.subtract(class_column, neighbor_column)
    else:
        neighbor_column = tf.gather(logistic_matrix, class_idx-1, axis=1)
        neighbor_column = tf.cast(neighbor_column, tf.float32)
        class_diff = tf.subtract(class_column, neighbor_column)
    return class_diff

def prob_true_class(class_idx, true_labels, logistic_matrix, num_classes):
    class_diff = prob_of_class(class_idx, logistic_matrix, num_classes)
    idx_for_class = tf.where(tf.equal(true_labels, class_idx))
    return tf.gather(class_diff, idx_for_class) + epsilon

def ordinal_logistic_loss(ordinal_vars, true_label, thresholds):
    """ Should maybe compute separately for each ordinal class """
    threshold_1 = tf.gather(thresholds, true_label)
    threshold_2 = tf.gather(thresholds, true_label-1)
    thresh_diff_1 = threshold_1 - ordinal_vars
    thresh_diff_2 = threshold_2 - ordinal_vars
    loss = tf.sigmoid(thresh_diff_1) - tf.sigmoid(thresh_diff_2)
    return tf.log(loss)

def find_vars_for_class(ord_class, ordinal_vars, true_labels):
    idx_for_class = tf.where(tf.equal(true_labels, ord_class))
    vars_for_class = tf.gather(ordinal_vars, idx_for_class)
    return vars_for_class

def model(inputs, labels, num_classes, num_features):
    ord_weights = get_ordinal_weights(num_features)
    init_thresholds = np.random.uniform(-1, 1, num_classes)
    init_thresholds = np.sort(init_thresholds)
    ord_thresholds = get_ordinal_thresholds(num_classes, init_thresholds)
    ord_vars = compute_ordinal_vars(ord_weights, inputs)
    logistic_matrix = tf.sigmoid(threshold_var_diff(ord_vars, ord_thresholds))
    total_log_prob = 0
    for ord_class in range(num_classes):
        probs_this_class = prob_true_class(ord_class, labels, logistic_matrix, num_classes)
        class_probs = tf.reduce_sum(tf.log(probs_this_class))
        total_log_prob = total_log_prob + class_probs
        
    return -total_log_prob

def run_model(inputs, labels, num_classes, num_features):
    log_loss = model(inputs, labels, num_classes, num_features)
    train_op = tf.train.AdamOptimizer(0.001).minimize(log_loss)
    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    sess=tf.Session(config=session_config)
    sess.run(tf.global_variables_initializer())
    num_iterations = 10000
    for i in range(num_iterations):
        _, iteration_loss = sess.run([train_op, log_loss])
        if i %5 == 0:
            print("Step {}, Loss: {}".format(i, iteration_loss))
    sess.close()

def main():
    data_filename = "example_ordinal_data.csv"
    X, y = load_sample_data(data_filename)
    num_classes = len(np.unique(y))
    num_examples, num_features = np.shape(X)
    run_model(X, y, num_classes, num_features)

if __name__ == "__main__":
    main()
