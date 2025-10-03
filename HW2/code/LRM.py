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
        self.assign_weights(np.random.randn(len(X[0]), self.k))

        labels = np.array(labels)
        y = np.eye(self.k)[labels.astype(int)]

        # Shuffling both x and y into a permutation of their indexes.
        perm = np.random.permutation(len(X))
        X = X[perm]
        y = y[perm]
        # For as many epochs as I wish
        for epoch in range(self.max_iter):
            # Split the shuffled inputs/outputs and split them into even sizes
            # calculate the gradient for each pair, add them up, divide by number
            # of inputs to get the average gradient. Subtract from weights at a 
            # factor of *learning_rate*. 
            for batch in range(0, len(X), batch_size):
                batch_X = X[batch : batch + batch_size]
                batch_y = y[batch : batch + batch_size]

                gradient = np.zeros_like(self.W)
                for input, output in zip(batch_X, batch_y):
                    gradient += self._gradient(input, output)

                gradient /= len(batch_X)

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
            _g: An array of shape [n_features, k_classes]. The gradient of
                cross-entropy with respect to self.W.
        """
        ### YOUR CODE HERE
        logits = self.softmax(self.W.T @ _x)
        # Reshape _x to column vector for outer product
        grad = np.outer(_x, (logits - _y))
        return grad
        ### END YOUR CODE

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        ### You must implement softmax by youself, otherwise you will not get credits for this part.

        ### YOUR CODE HERE
        # Numerically stable softmax
        x = x - np.max(x)
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)
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

        logits = X @ self.W  # shape: [n_samples, k]
        probs = np.apply_along_axis(self.softmax, 1, logits)
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
        preds = self.predict(X)
        return np.mean(preds == labels)
		### END YOUR CODE
    
    
    def assign_weights(self, weights):
        self.W = weights
        return self

