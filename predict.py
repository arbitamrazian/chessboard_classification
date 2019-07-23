from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf
import chess
from train import  get_estimator, ondisk_train_input_fn, collect_data

DIR = 'training_data/standard/'

def main(unused_argv):

    test_data_path = DIR+'test/'
    mnist_classifier = get_estimator()

    test_data = collect_data(test_data_path+'images/')
    eval_results = mnist_classifier.predict(
        input_fn=lambda: ondisk_train_input_fn(test_data))
    outputs = list(eval_results)

    num_accurate = 0
    for i in range(len(outputs)):
        y_hat = np.array(outputs[i])
        y = np.array(test_data[i][1])
        is_correct = int(np.array_equal(y_hat,y))
        if(is_correct==0):
            print(test_data[i],y_hat)
        num_accurate = num_accurate + is_correct
    
    print(num_accurate/len(outputs))

if __name__ == "__main__":
  tf.app.run()
