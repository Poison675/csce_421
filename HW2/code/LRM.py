#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 12:00:48 2019

@author: Aidan Veselka in part
"""

import numpy as np
import sys

"""This script implements a multi-class logistic regression model.
"""

class logistic_regression_multiclass(object):
	
    def __init__(self, learning_rate, max_iter, k):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.k = k 
        
    def fit_miniBGD(self, X, labels, batch_size):
        """Train perceptron model on data (X,y) with mini-Batch GD.

        Args:
            X: An array of shape [n_samples, n_features].
            labels: An array of shape [n_samples,].  Only contains 0,..,k-1.
            batch_size: An integer.

        Returns:
            self: Returns an instance of self.

        Hint: the labels should be converted to one-hot vectors, for example: 1----> [0,1,0]; 2---->[0,0,1].
        """

		### YOUR CODE HERE
        self.assign_weights(np.random.randn(3, len(X[0])))
        # My uncreative version:
        # y = np.zeros(shape=(labels, np.max(labels)))
        # for ind, output, label in enumerate(zip(y, labels)):
        #     output[label] = 1
        #     y[ind] = output

        # By god this is a pretty way of making a list of indexes one-hot. Credit to google.
        labels = np.array(labels)
        y = np.eye(self.k)[labels.astype(int)]

        # Shuffling both x and y into a permutation of their indexes.
        perm = np.random.permutation(len(X))
        X = X[perm]
        y = y[perm]
        # For as many epochs as I wish
        for epoch in range(1):
            # Split the shuffled inputs/outputs and split them into even sizes
            # calculate the gradient for each pair, add them up, divide by number
            # of inputs to get the average gradient. Subtract from weights at a 
            # factor of *learning_rate*. 
            for batch in range(0, len(X), batch_size):
                batch_X = X[batch : batch + batch_size]
                batch_y = y[batch : batch + batch_size]

                gradient = 0
                for input, output in zip(batch_X, batch_y):
                    gradient += self._gradient(input, output)

                gradient /= batch

                self.W += -self.learning_rate * gradient

        return self
		### END YOUR CODE
    

    def _gradient(self, _x, _y):
        """Compute the gradient of cross-entropy with respect to self.W
        for one training sample (_x, _y). This function is used in fit_*.

        Args:
            _x: An array of shape [n_features,].
            _y: One_hot vector. 

        Returns:
            _g: An array of shape [n_features,]. The gradient of
                cross-entropy with respect to self.W.
        """
		### YOUR CODE HERE
        logits = self.softmax(self.W.T @ _x)
        return logits - _y

		### END YOUR CODE
    
    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        ### You must implement softmax by youself, otherwise you will not get credits for this part.

		### YOUR CODE HERE
        # Uh. Idk what it means "by yourself" but this is the calculation based off the slide 5/16 T06-softmax-slides.
        # To clarify, I did come up with this myself. Although i cant imagine many peoples ideas will be that much
        # different than mine?
        return np.exp(x) / np.sum(np.exp(x))

		### END YOUR CODE
    
    def get_params(self):
        """Get parameters for this perceptron model.

        Returns:
            W: An array of shape [n_features,].
        """
        if self.W is None:
            print("Run fit first!")
            sys.exit(-1)
        return self.W


    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 0,..,k-1.
        """
		### YOUR CODE HERE
        probs = np.array(self.softmax(prob) for prob in X)
        return np.argmax(probs, axis=1)
		### END YOUR CODE


    def score(self, X, labels):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            labels: An array of shape [n_samples,]. Only contains 0,..,k-1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. labels.
        """
		### YOUR CODE HERE
        logits = self.predict(X)
        y = np.eye(self.k)[labels]
        return np.sum(np.where((logits == y), 1, 0)) / len(logits)
		### END YOUR CODE
    
    
    def assign_weights(self, weights):
        self.W = weights
        return self

