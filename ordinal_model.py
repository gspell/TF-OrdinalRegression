import numpy as np
import tensorflow as tf
import pandas as pd

from ordinal_eval import predict_labels, get_error_metric
epsilon = 1e-30

class OrdinalModel():
    def __init__(self, num_classes, num_features,
                 inputs, labels, threshold_init=None):
        self.num_classes = num_classes
        self.num_features = num_features
        self.inputs = inputs
        self.labels = labels
        
        self.ord_weights = self._get_ordinal_weights()
        self.ord_thresholds = self._get_ordinal_thresholds(init_vals = threshold_init)
        self.ordinal_vars = self._compute_ordinal_vars()
        self.logistic_matrix = self.compute_logistic_matrix()
        
    def _get_ordinal_weights(self):
        weights = tf.get_variable("ordinal_weights", shape = [self.num_features, 1])
        return weights

    def _get_ordinal_thresholds(self, init_vals = None):
        if init_vals is not None:
            """ Note these are constant, not tunable """
            ordinal_thresholds = tf.constant(init_vals, name="ord_thresholds")
        else:
            init_thresholds = np.random.uniform(-1, 1, self.num_classes)
            init_thresholds = np.sort(init_thresholds)
            ordinal_thresholds = tf.constant(init_thresholds, name="ord_thresholds")
            """ This is currently not suggested for use
            shape = [self.num_classes, 1]
            unordered_thresholds = tf.Variable(tf.random_uniform(shape, minval= -1,
                                                                 maxval = 1),
                                               name="ord_thresholds")
            ordinal_thresholds = tf.contrib.framework.sort(unordered_thresholds)
            """
        return ordinal_thresholds

    def _compute_ordinal_vars(self):
        inputs = tf.cast(self.inputs, tf.float32)
        ordinal_vars = tf.matmul(inputs, self.ord_weights,
                                 transpose_a=False, transpose_b=False)
        return ordinal_vars

    def _threshold_var_diff(self):
        """ The difference is accomplished via broadcasting.
        Does C_l - z_it for all thresholds """
        thresholds = tf.cast(self.ord_thresholds, tf.float32)
        ordinal_vars = tf.cast(self.ordinal_vars, tf.float32)
        differences = tf.subtract(thresholds, ordinal_vars)
        return differences

    def compute_logistic_matrix(self):
        return tf.sigmoid(self._threshold_var_diff())
                          
    def _class_prob(self, class_idx):
        class_column = tf.gather(self.logistic_matrix, class_idx, axis=1)
        class_column = tf.cast(class_column, tf.float32)
        if class_idx == self.num_classes -1:
            neighbor_column = 1.0
            class_diff = tf.subtract(neighbor_column, class_column)
        elif class_idx == 0:
            neighbor_column = 0.0
            class_diff = tf.subtract(class_column, neighbor_column)
        else:
            neighbor_column = tf.gather(self.logistic_matrix, class_idx-1, axis=1)
            neighbor_column = tf.cast(neighbor_column, tf.float32)
            class_diff = tf.subtract(class_column, neighbor_column)
        return class_diff + epsilon

    def _all_class_probs(self):
        probs_list = []
        for ord_class in range(self.num_classes):
            class_prob = self._class_prob(ord_class, self.logistic_matrix)
            probs_list.append(class_prob)
        all_class_probs = tf.stack(probs_list, axis=1)
        return all_class_probs

    def _true_class_prob(self, class_idx):
        class_prob = self._class_prob(class_idx)
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

def load_example_data(filename):
    """ Can also get Boston dataset from SKlearn, which might be good """
    data = pd.read_csv(filename, header=None)
    X = data[[0, 1, 2, 3]].values
    y = data[4].values
    return X, y

def run_model(X, y, num_classes, num_features):
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, num_features], name="inputs")
    labels = tf.placeholder(dtype=tf.int32, shape=[None], name="labels")
    model = OrdinalModel(num_classes, num_features, inputs=inputs, labels=labels)
    log_loss = model.total_loss()
    train_op = tf.train.AdamOptimizer(0.001).minimize(log_loss)
    session_config = tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    sess = tf.Session(config=session_config)
    sess.run(tf.global_variables_initializer())
    num_iterations = 1000
    for i in range(num_iterations):
        _, iteration_loss = sess.run([train_op, log_loss],
                                     feed_dict = {inputs:X, labels:y})
        if i % 10 == 0:
            print("Step {}, Loss: {}".format(i, iteration_loss))
    sess.close()
    
def main():
    data_filename = "example_ordinal_data.csv"
    X,y = load_example_data(data_filename)
    num_classes = len(np.unique(y))
    num_examples, num_features = np.shape(X)
    final_eval = run_model(X, y, num_classes, num_features)

if __name__ == "__main__":
    main()
