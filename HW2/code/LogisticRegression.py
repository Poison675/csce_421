import numpy as np
import sys

"""This script implements a two-class logistic regression model.
"""

class logistic_regression(object):
	
    def __init__(self, learning_rate, max_iter):
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def fit_BGD(self, X, y):
        """Train perceptron model on data (X,y) with Batch Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
		### YOUR CODE HERE
        self.fit_miniBGD(X, y, len(X))

		### END YOUR CODE
        return self

    def fit_miniBGD(self, X, y, batch_size):
        """Train perceptron model on data (X,y) with mini-Batch Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.
            batch_size: An integer.

        Returns:
            self: Returns an instance of self.
        """
		### YOUR CODE HERE
        weights = np.random.randn(3)
        self.assign_weights(weights)
        
        for epoch in range(5):
            indices = np.random.permutation(len(X))
            X = X[indices]
            y = y[indices]

            for i in range(0, len(X), batch_size):
                batch_x = X[i : i + batch_size]
                batch_y = y[i : i + batch_size]
                # This has selected a batch or a slice of the inputs from i to i + batch_size
                gradient = 0
                for input, output in zip(batch_x, batch_y):
                    gradient += self._gradient(input, output)
                
                # Find the average gradient of the batch
                gradient /= batch_size
                # Update
                self.W += -self.learning_rate * gradient

		### END YOUR CODE
        return self

    def fit_SGD(self, X, y):
        """Train perceptron model on data (X,y) with Stochastic Gradient Descent.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
		### YOUR CODE HERE
        weights = np.random.randn(3)
        self.assign_weights(weights)
        for epoch in range(5):
            indices = np.random.permutation(len(X))
            X = X[indices]
            y = y[indices]

            for input, output in zip(X, y):
                # Find the grad of each input WRT weights and subtract a fraction of it from the weights.
                self.W += -self.learning_rate * self._gradient(input, output) 

		### END YOUR CODE
        return self

    def _gradient(self, _x, _y):
        """Compute the gradient of cross-entropy with respect to self.W
        for one training sample (_x, _y). This function is used in fit_*.

        Args:
            _x: An array of shape [n_features,].
            _y: An integer. 1 or -1.

        Returns:
            _g: An array of shape [n_features,]. The gradient of
                cross-entropy with respect to self.W.
        """
		### YOUR CODE HERE
        # THIS SIMPLIFIES TO SIGMOID(y * W.T * X) * y * X
        ex = np.exp(-_y * np.dot(self.W, _x))
        return -(ex / (1 + ex)) * _y * _x
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

    def predict_proba(self, X):
        """Predict class probabilities for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds_proba: An array of shape [n_samples, 2].
                Only contains floats between [0,1].
        """
		### YOUR CODE HERE

        logits = np.dot(X, self.W)
        probs_pos = 1 / (1 + np.exp(-logits))
        probs_neg = 1 - probs_pos
        preds_proba = np.column_stack((probs_pos, probs_neg))

        return preds_proba

		### END YOUR CODE


    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 1 or -1.
        """
		### YOUR CODE HERE
        # My version:
        # preds = np.zeros(shape=(len(X),))
        # for i, input in enumerate(X):
        #     preds[i] = 1 if (np.dot(input, self.W) > 0) else -1

        # return preds
        
        # Using copilot to make it more pretty and concise. This is me giving credit where it is due.
        return np.where(np.dot(X, self.W) > 0, 1, -1)

		### END YOUR CODE

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. y.
        """
		### YOUR CODE HERE
        return sum(np.where((self.predict(X) == y), 1, 0)) / len(X)
		### END YOUR CODE
    
    def assign_weights(self, weights):
        self.W = weights
        return self

