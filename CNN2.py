# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 04:30:48 2019

@author: Dell
"""

import tensorflow as tf 

class CNN2:
    
        learning_rate = 0.001
        num_steps =1000
        batch_size = 128
        
        # Network Parameters
        num_input = 648 # img shape: 36*18
        num_classes = 2 # pedetrian or non-pedestrian
        dropout = 0.25 # Dropout, probability to drop a unit
        
        def __init__(self):
            self.model = tf.estimator.Estimator(self.model_fn)
            
        def conv_net(self, x_dict, n_classes, dropout, reuse, is_training):
            # Define a scope for reusing the variables
            with tf.variable_scope('ConvNet', reuse=reuse):
                # TF Estimator input is a dict, in case of multiple inputs
                x = x_dict['images']
                #(non-)pedestrian image 36*18
                # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
                x = tf.reshape(x, shape=[-1, 36, 18, 1])
        
                # Convolution Layer with 32 filters and a kernel size of 5
                conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
                # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
                conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
        
                # Convolution Layer with 64 filters and a kernel size of 3
                conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
                # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
                conv2 = tf.layers.max_pooling2d(conv2, 2, 2)
        
                # Flatten the data to a 1-D vector for the fully connected layer
                fc1 = tf.contrib.layers.flatten(conv2)
        
                # Fully connected layer (in tf contrib folder for now)
                fc1 = tf.layers.dense(fc1, 1024)
                # Apply Dropout (if is_training is False, dropout is not applied)
                fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)    
        
                # Output layer, class prediction
                out = tf.layers.dense(fc1, n_classes)
        
                return out
            
        def model_fn(self, features, labels, mode):
                # Build the neural network
                # Because Dropout have different behavior at training and prediction time, we
                # need to create 2 distinct computation graphs that still share the same weights.
                logits_train = self.conv_net(features, self.num_classes, self.dropout, reuse=False,
                                        is_training=True)
                logits_test = self.conv_net(features, self.num_classes, self.dropout, reuse=True,
                                       is_training=False)
            
                # Predictions
                pred_classes = tf.argmax(logits_test, axis=1)
                #pred_probas = tf.nn.softmax(logits_test)
            
                # If prediction mode, early return
                if mode == tf.estimator.ModeKeys.PREDICT:
                    return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)
            
                # Define loss and optimizer
                loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits_train, labels=tf.cast(labels, dtype=tf.int32)))
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                train_op = optimizer.minimize(loss_op,
                                              global_step=tf.train.get_global_step())
            
                # Evaluate the accuracy of the model
                acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)
            
                # TF Estimators requires to return a EstimatorSpec, that specify
                # the different ops for training, evaluating, ...
                estim_specs = tf.estimator.EstimatorSpec(
                    mode=mode,
                    predictions=pred_classes,
                    loss=loss_op,
                    train_op=train_op,
                    eval_metric_ops={'accuracy': acc_op})
            
                return estim_specs  
            
        def train(self, X1, y1):
            X1= X1 / 255.0
           # Define the input function for training
            input_fn = tf.estimator.inputs.numpy_input_fn(
                x={'images': X1.astype(float).reshape(29400, 36, 18, 1)}, y=y1.astype(float),
                batch_size=self.batch_size, num_epochs=5, shuffle=True)
            # Train the Model
            self.model.train(input_fn, steps=self.num_steps)
            
        def print_accuracy(self, X2, y2):
            X2= X2 / 255.0
            # Evaluate the Model
            # Define the input function for evaluating
            input_fn = tf.estimator.inputs.numpy_input_fn(
                x={'images': X2.astype(float).reshape(19600, 36, 18, 1)}, y=y2.astype(float),
                batch_size=self.batch_size, shuffle=False)
            # Use the Estimator 'evaluate' method
            e = self.model.evaluate(input_fn)
            
            print("CNN_accuracy:", e['accuracy'])