# Ordinal Regression

Ordinal regression concerns multi-label data in which the data labels are ordered relative to each other.  As a deep learning researcher, I've encountered a problem setting in which ordinal regression seems appropriate, but I haven't found a Tensorflow implementation of ordinal regression methods. This is my attempt to establish ordinal regression methods in Tensorflow so that I may apply it to my research.

## Ordinal Thresholds

As of right now, I have not found a way in Tensorflow to impose that the ordinal thresholds remain non-decreasing as they are tuned as parameters of the model via backpropagation.  In the interim, I've simply initialized the thresholds to a sorted non-decreasing random vector drawn from a uniform distribution over a particular range. These thresholds are treated as a constant. I've found that this severely handicaps the model, but it does allow the model to train.