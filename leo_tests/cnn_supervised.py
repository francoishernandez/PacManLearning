from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)

import gzip
import pickle
file = gzip.open("train_actions.gz", 'rb')
actionsLoad = pickle.load(file)
trainActions = np.asarray(actionsLoad)

file = gzip.open("train_states.gz", 'rb')
statesLoad = pickle.load(file)
trainStates = np.asarray(statesLoad)

batch_size = 10

def cnn_model_fn(features, labels, mode):
	#features = features.astype(dtype=np.float32)

	# Input Layer
    input_layer = tf.reshape(features, [batch_size, 121, 195, 1])
    input_layer = tf.to_float(input_layer)
    # Modèle simplifié avec un seul CNN

	# Conv Layer
    conv1 = tf.layers.conv2d(
		inputs=input_layer,
		filters=16,
         strides=(10,10),
		kernel_size=[10, 10],
		padding="same",
		activation=tf.nn.relu)

	# Dense Layer
    conv1_flat = tf.reshape(conv1, [batch_size, 4160])
    dense = tf.layers.dense(inputs=conv1_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
		inputs=dense, rate=0.4, training=mode == learn.ModeKeys.TRAIN)

	# Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=5)

    loss = None
    train_op = None

	# Calculate Loss (for both TRAIN and EVAL modes)
    if mode != learn.ModeKeys.INFER:
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=5)
        loss = tf.losses.softmax_cross_entropy(
                onehot_labels=onehot_labels, logits=logits)

	# Configure the Training Op (for TRAIN mode)
    if mode == learn.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(
                loss=loss,
                global_step=tf.contrib.framework.get_global_step(),
                learning_rate=0.001,
                optimizer="SGD")

	# Generate Predictions
    predictions = {
		"classes": tf.argmax(
			input=logits, axis=1),
		"probabilities": tf.nn.softmax(
			logits, name="softmax_tensor")
	}

	# Return a ModelFnOps object
    return model_fn_lib.ModelFnOps(
		mode=mode, predictions=predictions, loss=loss, train_op=train_op)





# Create the Estimator
pacman_classifier = learn.Estimator(
	model_fn=cnn_model_fn, model_dir="/tmp/convnet_model")


# Set up logging for predictions
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
	tensors=tensors_to_log, every_n_iter=50)

# Train the model
pacman_classifier.fit(
	x=trainStates,
	y=trainActions,
	batch_size=batch_size,
	steps=20000,
	monitors=[logging_hook])

# Configure the accuracy metric for evaluation
metrics = {
	"accuracy":
		learn.metric_spec.MetricSpec(
			metric_fn=tf.metrics.accuracy, prediction_key="classes"),
}

    
# Evaluate the model and print results
eval_results = pacman_classifier.evaluate(
    x=trainStates, y=trainActions, metrics=metrics)
print(eval_results)


