import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

epsilon = 1e-30

class MultiClassRegressionModel():
    def __init__(self, num_classes, num_features,
                 inputs, labels):
        self.num_classes = num_classes
        self.num_features = num_features
        self.inputs = inputs
        self.labels = labels
        
        self.logits = self._map_to_logits(self.inputs)
        self.loss = self.total_loss()
        self.error_metric = self.get_error_metric()
        self.accuracy = self.get_accuracy()
    
    def _map_to_logits(self, input_features):
        """ Could add a non-linearity here """
        hidden_weights = tf.get_variable("hidden_weights",
                                         shape=[self.num_features, self.num_classes])
        hidden_biases = tf.get_variable("hidden_biases", shape=[self.num_classes])
        outputs = tf.nn.xw_plus_b(input_features, hidden_weights, hidden_biases)
        outputs = tf.nn.relu(outputs)
        return outputs


    def _predicted_labels(self):
        predicted_labels = tf.argmax(self.logits, axis=1, name="predicted_labels")
        return tf.cast(predicted_labels, tf.int32)

    def total_loss(self):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                       labels=self.labels)
        return tf.reduce_mean(cross_entropy)


    def get_error_metric(self, metric="MAE"):
        errors = tf.cast(tf.subtract(self.labels, self._predicted_labels()), tf.float32)
        if metric == "MAE":
            metric = tf.abs(errors)
        elif metric == "MSE":
            metric = tf.square(errors)
        return tf.reduce_mean(metric)

    def get_accuracy(self):
        correct_predictions = tf.equal(self._predicted_labels(), self.labels)
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))
        return accuracy
    
def configure_session():
    session_config = tf.ConfigProto(allow_soft_placement=True,
                                    log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    return tf.Session(config=session_config)

def get_feedDicts(placeholders, train_data, val_data):
    input_legs, input_days, labels = placeholders
    train_feed_dict = {input_legs: train_data[0], input_days: train_data[1],
                       labels: train_data[2]}
    val_feed_dict = {input_legs: val_data[0], input_days: val_data[1],
                     labels: val_data[2] }
    return train_feed_dict, val_feed_dict

def split_data(X, y, percent_test):
    num_holdout = int(percent_test * len(y))
    permutation = np.arange(len(y))
    np.random.shuffle(permutation) # does in place
    shuffled_X, shuffled_y = X[permutation], y[permutation]
    train_X, test_X = shuffled_X[:-num_holdout], shuffled_X[-num_holdout:]
    train_y, test_y = shuffled_y[:-num_holdout], shuffled_y[-num_holdout:]
    print("Indices for test: {}".format(permutation[-num_holdout:]))
    return train_X, train_y, test_X, test_y

def run_model(X, y, num_classes, num_features):
    tf.set_random_seed(seed=0)
    np.random.seed(seed=0)
    train_X, train_y, test_X, test_y = split_data(X, y, percent_test=0.2)
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, num_features], name="inputs")
    labels = tf.placeholder(dtype=tf.int32, shape=[None], name="labels")
    model = MultiClassRegressionModel(num_classes, num_features, inputs, labels)
    
    log_loss = model.loss
    mae = model.error_metric
    train_op = tf.train.AdamOptimizer(0.001).minimize(log_loss)
    sess = configure_session()
    sess.run(tf.global_variables_initializer())
    num_iterations=5000
    for i in range(num_iterations):
        _, iteration_loss = sess.run([train_op, log_loss],
                                     feed_dict = {inputs:train_X, labels:train_y})
        if i % 10 == 0:
            training_mae = sess.run(mae, feed_dict= {inputs:train_X, labels:train_y})
            test_mae = sess.run(mae, feed_dict = {inputs:test_X, labels:test_y})
            print("Step {}, Train Loss: {:.4f}, Train MAE: {:.3f}, Test MAE: {:.3f}".format(
                   i, iteration_loss, training_mae, test_mae))
    sess.close()
   
def load_example_data(filename):
    """ Can also get Boston dataset from SKlearn, which might be good """
    data = pd.read_csv(filename, header=None)
    X = data[[0, 1, 2, 3]].values
    y = data[4].values - 1 # Need to 0-index class labels
    return X, y

def main():
    data_filename = "example_ordinal_data.csv"
    X, y = load_example_data(data_filename)
    num_classes = len(np.unique(y))
    num_examples, num_features = np.shape(X)
    run_model(X, y, num_classes, num_features)

if __name__ == "__main__":
    main()
