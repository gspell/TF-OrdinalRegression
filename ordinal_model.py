import numpy as np
import tensorflow as tf
import pandas as pd
import tensorflow.contrib.slim as slim

from sklearn.datasets import load_boston
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from ordinal_eval import predict_labels, get_error_metric
epsilon = 1e-30
INF = 1e30

class OrdinalModel():
    def __init__(self, num_classes, num_features,
                 inputs, labels, threshold_init=None):
        self.num_classes = num_classes
        self.num_features = num_features
        self.inputs = inputs
        self.labels = labels
        
        self.ord_weights = self._get_ordinal_weights()
        self.ord_thresholds = self._get_ordinal_thresholds(init_vals = threshold_init)
        print("Ordinal Thresholds shape {}".format(self.ord_thresholds.get_shape()))
        self.ordinal_vars = self._compute_ordinal_vars()
        print("Ordinal vars shape {}".format(self.ordinal_vars.get_shape()))
        self.sigmoid_matrix = self._sigmoid_over_classDiffs()
        
    def _get_ordinal_weights(self):
        weights = tf.get_variable("ordinal_weights", shape = [self.num_features, 1])
        return weights

    def _get_ordinal_thresholds(self, init_vals = None):
        if init_vals is not None:
            """ Note these are constant, not tunable """
            ordinal_thresholds = tf.constant(init_vals, name="ord_thresholds")
        else:
            init_thresholds = np.random.uniform(-10, 10, self.num_classes-1)
            init_thresholds = np.sort(init_thresholds)
            print("Initial threshold values: {}".format(init_thresholds))
            thresholds = []
            temp_threshold = tf.get_variable("neg_inf_thresh", initializer=-INF,
                                             trainable=False)
            thresholds.append(tf.cast(temp_threshold, tf.float64))
            for i in range(self.num_classes-1):
                threshold_name = "threshold_"+str(i)
                #constraint = lambda x: tf.clip_by_value(x-thresholds[i], 0.001, np.infty)
                temp_threshold = tf.get_variable(threshold_name,
                                                 initializer=init_thresholds[i],
                                                 trainable=True)
                thresholds.append(temp_threshold)
            temp_threshold = tf.get_variable("inf_thresh", initializer=INF,
                                             trainable = False)
            thresholds.append(tf.cast(temp_threshold, tf.float64))
            ordinal_thresholds = tf.stack(thresholds)
        return ordinal_thresholds

    def _compute_ordinal_vars(self):
        """ Could add a non-linearity and biases here """
        inputs = tf.cast(self.inputs, tf.float32)
        """
        hidden_weights_1 = tf.get_variable("hidden_weights_1", shape=[self.num_features, 2])
        hidden_biases_1 = tf.get_variable("hidden_biases_1", shape=[2])
        ordinal_vars = tf.nn.xw_plus_b(inputs, hidden_weights_1, hidden_biases_1)
        ordinal_vars = tf.nn.relu(ordinal_vars)
        hidden_weights_2 = tf.get_variable("hidden_weights_2", shape=[2, 1])
        hidden_biases_2 = tf.get_variable("hidden_biases_2", shape=[1])
        """
        ordinal_vars = slim.fully_connected(inputs, 4)
        ordinal_vars = slim.fully_connected(inputs, 2)
        ordinal_vars = slim.fully_connected(ordinal_vars, 1, activation_fn=None) 
        return ordinal_vars

    def _threshold_var_diff(self):
        """ The difference is accomplished via broadcasting.
        Does C_l - z_it for all thresholds """
        thresholds = tf.cast(self.ord_thresholds, tf.float32)
        ordinal_vars = tf.cast(self.ordinal_vars, tf.float32)
        differences = tf.subtract(thresholds, ordinal_vars)
        return differences

    def _sigmoid_over_classDiffs(self):
        return tf.sigmoid(self._threshold_var_diff())
                          
    def _single_class_prob(self, class_idx):
        class_column = tf.gather(self.sigmoid_matrix, class_idx+1, axis=1)
        class_column = tf.cast(class_column, tf.float32)

        neighbor_column = tf.gather(self.sigmoid_matrix, class_idx, axis=1)
        neighbor_column = tf.cast(neighbor_column, tf.float32)
        class_prob = tf.subtract(class_column, neighbor_column)
        return class_prob + epsilon

    def _all_class_probs(self):
        probs_list = []
        for ord_class in range(self.num_classes):
            class_prob = self._single_class_prob(ord_class)
            probs_list.append(class_prob)
        all_class_probs = tf.stack(probs_list, axis=1)
        return all_class_probs

    def _true_class_prob(self, class_idx):
        class_prob = self._single_class_prob(class_idx)
        idx_for_class = tf.where(tf.equal(self.labels, class_idx))
        return tf.gather(class_prob, idx_for_class)

    def _single_class_log_prob(self, class_idx):
        class_probs = self._true_class_prob(class_idx)
        return tf.reduce_sum(tf.log(class_probs))

    def total_loss(self):
        """ Negative log-likelihood """
        total_log_prob = 0.0
        for ord_class in range(self.num_classes):
            log_prob = self._single_class_log_prob(ord_class)
            total_log_prob = total_log_prob + log_prob
        return -total_log_prob

    def _predicted_labels(self):
        predicted_labels = tf.argmax(self._all_class_probs(), axis=1)
        return tf.cast(predicted_labels, tf.int32)

    def _get_error_metric(self, metric="MAE"):
        errors = tf.cast(tf.subtract(self.labels, self._predicted_labels()), tf.float32)
        if metric == "MAE":
            metric = tf.abs(errors)
        elif metric == "MSE":
            metric = tf.square(errors)
        return tf.reduce_mean(metric)

    def get_accuracy(self):
        correct_predictions = tf.equal(self._predicted_labels(), self.labels)
        accuracy = tf.reduce_mean(tf.cast(correct_predcitions, "float"))
        return accuracy
   
def load_example_data(filename):
    """ Can also get Boston dataset from SKlearn, which might be good """
    data = pd.read_csv(filename, header=None)
    X = data[[0, 1, 2, 3]].values
    y = data[4].values - 1 # Need to 0-index class labels
    return X, y

def load_boston_data():
    X, y = load_boston(return_X_y=True)
    y = np.round(y).astype(int)
    return X, y

def split_data(X, y, percent_test):
    num_holdout = int(percent_test * len(y))
    permutation = np.arange(len(y))
    np.random.shuffle(permutation) # does in place
    shuffled_X, shuffled_y = X[permutation], y[permutation]
    train_X, test_X = shuffled_X[:-num_holdout], shuffled_X[-num_holdout:]
    train_y, test_y = shuffled_y[:-num_holdout], shuffled_y[-num_holdout:]
    print("Indices for test: {}".format(permutation[-num_holdout:]))
    return train_X, train_y, test_X, test_y

def run_model(X, y, num_classes, num_features, seed=0):
    tf.set_random_seed(seed=seed)
    np.random.seed(seed=seed)
    train_X, train_y, test_X, test_y = split_data(X, y, percent_test = 0.2)
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, num_features], name="inputs")
    labels = tf.placeholder(dtype=tf.int32, shape=[None], name="labels")
    model = OrdinalModel(num_classes, num_features, inputs=inputs, labels=labels)
    log_loss = model.total_loss()
    mae = model._get_error_metric(metric="MAE")
    train_op = tf.train.AdamOptimizer(0.001).minimize(log_loss)
    session_config = tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    sess = tf.Session(config=session_config)
    sess.run(tf.global_variables_initializer())
    num_iterations = 5000
    for i in range(num_iterations):
        _, iteration_loss = sess.run([train_op, log_loss],
                                     feed_dict = {inputs:train_X, labels:train_y})
        if i % 20 == 0:
            train_mae = sess.run(mae, feed_dict={inputs:train_X, labels:train_y})
            test_mae = sess.run(mae, feed_dict = {inputs:test_X, labels:test_y})
            print("Step {}, Loss: {:.4f}, Train MAE: {:.3f}, Test MAE: {:.3f}".format(
                  i, iteration_loss, train_mae, test_mae))
            #ord_thresholds = sess.run(model.ord_thresholds)
            #print("Ordinal Thresholds: {}".format(ord_thresholds))
    sess.close()
    
def main():
    data_filename = "example_ordinal_data.csv"
    #X, y = load_boston_data()
    X,y = load_example_data(data_filename)
    num_classes = len(np.unique(y))
    num_examples, num_features = np.shape(X)
    final_eval = run_model(X, y, num_classes, num_features)

if __name__ == "__main__":
    main()
