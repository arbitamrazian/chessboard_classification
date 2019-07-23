from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import csv
from PIL import Image
from sklearn.model_selection import train_test_split
from os import listdir
from os.path import isfile, join

DIR = 'training_data/standard/'

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  input_layer = tf.reshape(features["image"], [-1, 180, 180, 1])

  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=64,
      kernel_size=[15, 15],
      padding="same",
      activation=tf.nn.relu)
  print(conv1.shape)

  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2) #90x90x32
  print(pool1.shape)

  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[15, 15],
      padding="same",
      activation=tf.nn.relu)
  print(conv2.shape)

  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2) #45x45x64
  print(pool2.shape)

  conv3 = tf.layers.conv2d(
      inputs=pool2,
      filters=32,
      kernel_size=[15, 15],
      padding="same",
      activation=tf.nn.relu)
  print(conv3.shape)
##
  pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
  print(pool3.shape)

  pool2_flat = tf.reshape(pool3, [-1, 22*22* 32])

  print(pool2_flat.shape)

  dense = tf.layers.dense(inputs=pool2_flat, units=512, activation=tf.nn.relu)
  print(dense.shape)

  dropout = tf.layers.dropout(
      inputs=dense, rate=0.3, training=mode == tf.estimator.ModeKeys.TRAIN)

  dense2 = tf.layers.dense(inputs=dropout, units=256, activation=tf.nn.relu)
  print(dense2.shape)

  dropout2 = tf.layers.dropout(
      inputs=dense2, rate=0.3, training=mode == tf.estimator.ModeKeys.TRAIN)

  logits = tf.layers.dense(inputs=dropout2, units=64)

  predictions = {
      "classes_probs": tf.sigmoid(logits,name='probs'),
      "prediction": tf.round(tf.sigmoid(logits),name='preds'),
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions['prediction'])
  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits( logits=logits, labels=labels), axis=1))

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, 
          predictions=predictions["prediction"]
      )
    }
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def get_estimator():
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="chess/mnist_convnet_model_v3_2")
    return mnist_classifier

def _ondisk_parse_(filename,label):
    label = tf.cast(label,tf.float32)
    image_string = tf.read_file(filename)
    image = tf.image.decode_png(image_string)
    image = tf.image.per_image_standardization(image)
    image = tf.cast(image,tf.int8)
    image = tf.cast(image,tf.float32)
    return (dict({'image':image}),label)

def ondisk_train_input_fn(data,batch_size=32):
    filenames = tf.constant([x[0] for x in data])
    labels = tf.constant([x[1] for x in data])
    dataset  = tf.data.Dataset.from_tensor_slices((filenames,labels))
    dataset = dataset.map(_ondisk_parse_).shuffle(True).batch(batch_size)
    ondisk_iterator = dataset.make_one_shot_iterator()
    return ondisk_iterator.get_next()

def get_labels_from_filename(f):
    output = []
    label = f.split('_')[0]
    output = [0 if v=='x' else 1 for v in label]
    return output

def collect_data(path):
    data = [(path+f,get_labels_from_filename(f)) for f in listdir(path) if isfile(join(path, f))]
    return data

def train(estimator):
    data_path = DIR+'train/'
    data = collect_data(data_path+'images/')
    tensors_to_log = {"prediction": "probs"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=100)

    estimator.train(
        input_fn= lambda:ondisk_train_input_fn(data,64),
        steps=10000,
        hooks=[logging_hook])

def validate(estimator):
    data_path = DIR+'validation/'
    data = collect_data(data_path+'images/')
    eval_results = estimator.evaluate(
	input_fn=lambda: ondisk_train_input_fn(data,64))
    print(eval_results)

def main(unused_argv):
    chess_classifier = get_estimator()
    train(chess_classifier)
    validate(chess_classifier)


if __name__ == "__main__":
  tf.app.run()
