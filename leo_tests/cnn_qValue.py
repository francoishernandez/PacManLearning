from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

tf.logging.set_verbosity(tf.logging.INFO)

# On récupère la replay memory initiale de la forme (s,a,r,s') et l'ensemble des
# états s seuls (fichiers créés au préalable par data_preprocessing.py) 
import gzip
import pickle
file = gzip.open("train_s_a_r_sBis.gz", 'rb')
replayMemory = pickle.load(file)

file = gzip.open("train_states.gz", 'rb')
statesLoad = pickle.load(file)
trainStates = np.asarray(statesLoad)

# Facteur de dévaluation
gamma = 0.5

# Cette fonction recalcule les Q-values cibles sur la replay memory
# A appeler après chaque nouvelle expérimentation lors du reinforcement
def recalculate_target(memory) :
    targets = []
    for i in range(len(memory)) :
        s, a, r, sPrime = memory[i]
        if (sPrime == "Final") :
            targets[i] = r
        else :
            MaxQvalue = cnn_model_fn(sPrime, None, learn.ModeKeys.INFER)["MaxQvalue"]
            targets[i] = r + gamma * MaxQvalue
    
    return targets

# Modèle de notre réseau de neurones
def cnn_model_fn(features, labels, mode):

	# Input Layer
	input_layer = tf.reshape(features, [-1, 84, 84, 1])

    # Modèle simplifié avec un seul CNN

	# Conv Layer
	conv1 = tf.layers.conv2d(
		inputs=input_layer,
		filters=32,
         strides=5,
		kernel_size=[10, 10],
		padding="same",
		activation=tf.nn.relu)

	# Dense Layer
	conv1_flat = tf.reshape(conv1, [-1, 15 * 15 * 64])
	dense = tf.layers.dense(inputs=conv1_flat, units=1024, activation=tf.nn.relu)
	dropout = tf.layers.dropout(
		inputs=dense, rate=0.4, training=mode == learn.ModeKeys.TRAIN)

	# Logits Layer
	logits = tf.layers.dense(inputs=dropout, units=5)

	loss = None
	train_op = None

	# Loss (régression sur les Q-values donc mean squared)
	if mode != learn.ModeKeys.INFER:
        	loss = tf.reduce_mean(tf.squared_difference(
                    tf.cast(logits, tf.int32), 
                    tf.cast(labels, tf.int32)))

	# Configure the Training Op (for TRAIN mode)
	if mode == learn.ModeKeys.TRAIN:
        	train_op = tf.contrib.layers.optimize_loss(
        		loss=loss,
        		global_step=tf.contrib.framework.get_global_step(),
        		learning_rate=0.01,
        		optimizer="SGD")

	# Generate Predictions
	predictions = {
		"action": tf.argmax(
			input=logits, axis=1),
		"probabilities": tf.nn.softmax(
			logits, name="softmax_tensor"),
         "MaxQvalue" : tf.reduce_max(
			input=logits)
	}

	# Return a ModelFnOps object
	return model_fn_lib.ModelFnOps(
		mode=mode, predictions=predictions, loss=loss, train_op=train_op)




# Création de l'estimateur
Qvalue_regressor = learn.Estimator(
	model_fn=cnn_model_fn, model_dir="/tmp/convnet_model")

# affichage log des prédictions
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
	tensors=tensors_to_log, every_n_iter=50)


# DEEP Q LEARNING

# Nombre d'itérations
imax = 1000
# Taille du sample du replayMemory
replayMemorySample = 100


for i in range(imax) :
    
    # Pour le reinforcement learning, ici devrait se situer le choix du coup à l'aide de :
    # action = cnn_model_fn(currentState, None, learn.ModeKeys.INFER)["action"]
    # On effectue ensuite l'action, on observe la récompense, et on peux ainsi
    # l'ajouter à la replay memory.
    
    targetQvalues = recalculate_target(replayMemory)

    # Train the model
    Qvalue_regressor.fit(
        x=stateFeatures,
        y=targetQvalues,
        batch_size=replayMemorySample,
        steps=1,
        monitors=[logging_hook])






## Configure the accuracy metric for evaluation
#metrics = {
#	"accuracy":
#		learn.metric_spec.MetricSpec(
#			metric_fn=tf.metrics.accuracy, prediction_key="classes"),
#}
#    
## Evaluate the model and print results
#eval_results = mnist_classifier.evaluate(
#    x=eval_data, y=eval_labels, metrics=metrics)
#print(eval_results)