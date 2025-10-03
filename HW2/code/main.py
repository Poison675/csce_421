import os
import matplotlib.pyplot as plt
from LogisticRegression import logistic_regression
from LRM import logistic_regression_multiclass
from DataReader import *

data_dir = "../data/"
train_filename = "training.npz"
test_filename = "test.npz"
    
def visualize_features(X, y):
    '''This function is used to plot a 2-D scatter plot of training features. 

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 1 or -1.

    Returns:
        No return. Save the plot to 'train_features.*' and include it
        in submission.
    '''
    ### YOUR CODE HERE
	# X shape: [n_samples, 2]
	# Want to convert shape to hold two dimensions i can plot on a graph for each point. X[0] will be 
	# the intensity of each sample, X[1] will be the symmetry of each sample.
    X = X.T
    plt.scatter(X[0], X[1], c=y, cmap='viridis', s=50, alpha=0.6)

    # Add labels and title
    plt.xlabel('Intensity')
    plt.ylabel('Symmetry')
    plt.title('Scatterplot of Data Colored by Label')

    # Save plot
    plt.savefig('scatterplot.png')
    plt.close()
    # For each pair (X[1, i], X[2, i]), it will be colored according to the label y[i].


    ### END YOUR CODE

def visualize_result(X, y, W):
    '''This function is used to plot a 2-D scatter plot of training features. 

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 1 or -1.

    Returns:
        No return. Save the plot to 'train_features.*' and include it
        in submission.
    '''
    ### YOUR CODE HERE
	# X shape: [n_samples, 2]
	# Want to convert shape to hold two dimensions i can plot on a graph for each point. X[0] will be 
	# the intensity of each sample, X[1] will be the symmetry of each sample.
    X = X.T
    scatter = plt.scatter(X[0], X[1], c=y, cmap='viridis', s=50, alpha=0.6)
    b, w1, w2 = W
    x2 = -(w1 * X[1] + b) / w2
    plt.plot(X[1], x2, "b-")

    # Add labels and title
    plt.xlabel('Intensity')
    plt.ylabel('Symmetry')
    plt.title('Scatterplot of Data Colored by Label')
    # Save plot
    plt.savefig('train_features.png')
    plt.close()
    # For each pair (X[1, i], X[2, i]), it will be colored according to the label y[i].def visualize_result(X, y, W):


def visualize_result_multi(X, y, W):
    '''This function is used to plot the softmax model after training. 

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 0,1,2.
        W: An array of shape [n_features, 3].
    
    Returns:
        No return. Save the plot to 'train_result_softmax.*' and include it
        in submission.
    '''
    ### YOUR CODE HERE
    # X shape: [n_samples, 2]
    # Scatter plot of two features colored by label
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50, alpha=0.6)

    # Plot decision boundaries for each pair of classes
    # W shape: [n_features, 3] (assuming bias is included as first feature)
    # For each pair of classes, plot the line where their scores are equal
    xx = np.linspace(-1, 0.5, 200)

    # If W includes bias as first row, features as next rows
    # For each pair of classes (i, j), plot the boundary where W[:,i].T @ [1, x1, x2] = W[:,j].T @ [1, x1, x2]
    colors = ['r', 'g', 'b']
    for i in range(W.shape[1]):
        for j in range(i+1, W.shape[1]):
            # W[:,i] and W[:,j]
            # w0, w1, w2 for each class
            w_i = W[:, i]
            w_j = W[:, j]
            # Decision boundary: (w0_i - w0_j) + (w1_i - w1_j)*x1 + (w2_i - w2_j)*x2 = 0
            # Solve for x2 as a function of x1
            if (w_j[2] - w_i[2]) != 0:
                x2_boundary = -(w_i[0] - w_j[0] + (w_i[1] - w_j[1]) * xx) / (w_i[2] - w_j[2])
                plt.plot(xx, x2_boundary, color=colors[i], linestyle='--', label=f'Boundary {i} vs {j}')

    plt.xlabel('Intensity')
    plt.ylabel('Symmetry')
    plt.title('Softmax Multiclass Logistic Regression Results')
    plt.legend()
    plt.savefig('train_result_softmax.png')
    plt.close()
    ### END YOUR CODE
    # For each pair (X[1, i], X[2, i]), it will be colored according to the label y[i].
	### END YOUR CODE

def main():
    # ------------Data Preprocessing------------
    # Read data for training.

    raw_data, labels = load_data(os.path.join(data_dir, train_filename))
    raw_train, raw_valid, label_train, label_valid = train_valid_split(raw_data, labels, 2300)

    ##### Preprocess raw data to extract features
    train_X_all = prepare_X(raw_train)
    valid_X_all = prepare_X(raw_valid)
    ##### Preprocess labels for all data to 0,1,2 and return the idx for data from '1' and '2' class.
    train_y_all, train_idx = prepare_y(label_train)
    valid_y_all, val_idx = prepare_y(label_valid)  

    ####### For binary case, only use data from '1' and '2'  
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    ####### Only use the first 1350 data examples for binary training. 
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx]
    ####### set lables to  1 and -1. Here convert label '2' to '-1' which means we treat data '1' as postitive class. 
    ### YOUR CODE HERE
    train_y[np.where(train_y==2)] = -1
    valid_y[np.where(valid_y==2)] = -1
    ### END YOUR CODE
    data_shape = train_y.shape[0] 

       # Visualize training data.
    input('Next graph: Visualize training set...')
    visualize_features(train_X[:, 1:3], train_y)

    # ------------Logistic Regression Sigmoid Case------------

    ##### Check BGD, SGD, miniBGD
    lrm = logistic_regression(learning_rate=0.5, max_iter=100)

    print("------- Training LRM in various ways: BGD, MiniBDG with full size, SGD, MiniBGD size 1, MiniBGD size 10 --------\n")
    lrm.fit_BGD(train_X, train_y)
    print(lrm.get_params())
    print(lrm.score(train_X, train_y))

    lrm.fit_miniBGD(train_X, train_y, data_shape)
    print(lrm.get_params())
    print(lrm.score(train_X, train_y))

    lrm.fit_SGD(train_X, train_y)
    print(lrm.get_params())
    print(lrm.score(train_X, train_y))

    lrm.fit_miniBGD(train_X, train_y, 1) # PRINTING ERROR FOR EACH BATCH? CHECK FOR VALID LOOP
    print(lrm.get_params())
    print(lrm.score(train_X, train_y))

    lrm.fit_miniBGD(train_X, train_y, 10)
    print(lrm.get_params())
    print(lrm.score(train_X, train_y))

    print('\n\n')

# Explore different hyper-parameters.
    ### YOUR CODE HERE
    print("------- Training LRM with various hyper-parameters --------\n")
    for lr in (0.25, 0.5, 1, 2, 5, 10):
        lrm.learning_rate = lr
        for epochs in (5, 10, 50, 100, 200):
            lrm.max_iter = epochs
            lrm.fit_miniBGD(train_X, train_y, 10)
            print(f'LR: {lr}, Epochs: {epochs}')
            print(lrm.get_params(), lrm.score(train_X, train_y))
		
    print('\n\n')
    ### END YOUR CODE

    # Visualize the your 'best' model after training.

    ### YOUR CODE HERE
    input('Next graph: LRM best model hyper-parameters...')
    lrm.learning_rate = 0.5
    lrm.max_iter = 50
    lrm.fit_miniBGD(train_X, train_y, 10)
    visualize_result(train_X[:, 1:3], train_y, lrm.get_params())
    ### END YOUR CODE

    # Use the 'best' model above to do testing. Note that the test data should be loaded and processed in the same way as the training data.
    ### YOUR CODE HERE
    input('Next graph: LRM best model validaiton set...')
    print(lrm.score(valid_X, valid_y))
    visualize_result(valid_X[:, 1:3], valid_y, lrm.get_params())
    
    print('\n\n')
    ### END YOUR CODE


   
    # ------------Logistic Regression Multiple-class case, let k= 3------------
    ###### Use all data from '0' '1' '2' for training
    train_X = train_X_all
    train_y = train_y_all
    valid_X = valid_X_all
    valid_y = valid_y_all

    #########  miniBGD for multiclass Logistic Regression
    print("------- Training LRM-multiclass --------\n")
    lrmMulti = logistic_regression_multiclass(learning_rate=0.5, max_iter=100, k=3)
    lrmMulti.fit_miniBGD(train_X, train_y, 10)
    print(lrmMulti.get_params())
    print(lrmMulti.score(train_X, train_y))

    # Explore different hyper-parameters.
    ### YOUR CODE HERE
    print("------- Training LRM-multiclass with various hyper-parameters --------\n")
    for lr in (0.25, 0.5, 1, 2, 5, 10):
        lrmMulti.learning_rate = lr
        for epochs in (5, 10, 50, 100, 200):
            lrmMulti.max_iter = epochs
            lrmMulti.fit_miniBGD(train_X, train_y, 10)
            print(f'LR: {lr}, Epochs: {epochs}')
            print(lrmMulti.get_params(), lrmMulti.score(train_X, train_y))

    print('\n\n')
    ### END YOUR CODE

    # Visualize the your 'best' model after training.
    input('Next graph: LRM-multiclass best model hyper-parameters...')
    lrmMulti.learning_rate = 0.5
    lrmMulti.max_iter = 50
    lrmMulti.fit_miniBGD(train_X, train_y, 10)
    visualize_result_multi(train_X[:, 1:3], train_y, lrmMulti.get_params())


    # Use the 'best' model above to do testing.
    ### YOUR CODE HERE
    input('Next graph: LRM-multiclass best model validation set...')
    lrmMulti.fit_miniBGD(valid_X_all, valid_y_all, 10)
    visualize_result_multi(valid_X_all[:, 1:3], valid_y_all, lrmMulti.get_params())
    
    print('\n\n')
    ### END YOUR CODE

    
    # ------------Connection between sigmoid and softmax------------
    ############ Now set k=2, only use data from '1' and '2' 

    #####  set labels to 0,1 for softmax classifer
    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx] 
    train_y[np.where(train_y==2)] = 0
    valid_y[np.where(valid_y==2)] = 0  

    ###### First, fit softmax classifer until convergence, and evaluate 
    ##### Hint: we suggest to set the convergence condition as "np.linalg.norm(gradients*1./batch_size) < 0.0005" or max_iter=10000:
    ### YOUR CODE HERE
    input('Next graph: LRM-multiclass max iterations...')
    lrmMulti = logistic_regression_multiclass(learning_rate=0.5, max_iter=1000, k=2)
    lrmMulti.fit_miniBGD(train_X, train_y, batch_size=10)
    print(lrmMulti.get_params(), lrmMulti.score(train_X, train_y))
    visualize_result_multi(train_X[:, 1:3], train_y, lrmMulti.get_params())
    
    print('\n\n')
    ### END YOUR CODE






    train_X = train_X_all[train_idx]
    train_y = train_y_all[train_idx]
    train_X = train_X[0:1350]
    train_y = train_y[0:1350]
    valid_X = valid_X_all[val_idx]
    valid_y = valid_y_all[val_idx] 
    #####       set lables to -1 and 1 for sigmoid classifer
    ### YOUR CODE HERE
    train_y[np.where(train_y==2)] = -1
    valid_y[np.where(valid_y==2)] = -1
    ### END YOUR CODE 

    ###### Next, fit sigmoid classifer until convergence, and evaluate
    ##### Hint: we suggest to set the convergence condition as "np.linalg.norm(gradients*1./batch_size) < 0.0005" or max_iter=10000:
    ### YOUR CODE HERE

    input('Next graph: LRM max iterations...')
    lrm = logistic_regression(learning_rate=0.5, max_iter=1000)
    lrm.fit_miniBGD(train_X, train_y, 10)
    print(lrm.get_params(), lrm.score(train_X, train_y))
    visualize_result(train_X[:, 1:3], train_y, lrm.get_params())
    
    print('\n\n')
    ### END YOUR CODE


    ################Compare and report the observations/prediction accuracy


    # ------------End------------


if __name__ == '__main__':
	main()
    
    
