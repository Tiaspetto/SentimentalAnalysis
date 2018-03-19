import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

import numpy as np
import matplotlib.pyplot as plt
import pickle

def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    
    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A, cache

def relu(Z):
    """
    Implement the RELU function.
    Arguments:
    Z -- Output of the linear layer, of any shape
    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache


def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.
    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently
    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.
    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently
    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ


def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    
    np.random.seed(1)
    parameters = {}
    L = len(layer_dims)            # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1]) #*0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        
    return parameters

def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.
    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    
    Z = W.dot(A)  + b
    
    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)
    
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
	"""
	Implement the forward propagation for the LINEAR -> ACTIVATION layer

	Arguments:
	A_prev -- activation from previous layer (or input data): (size of previous layer, number of examples)
	W 	   -- weight matrix: numpy array of shape (size of current layer, size of previous layer)
	b 	   -- bias vector, numpy array of shape （size of the current layer, 1）
	activation -- the activation to be uesd in this layer, stored as a text string: "sigmoid" or "relu"

	Return:
	A 		-- the output of the activation function, also called the post-activation on value
	cache
	"""
	Z, linear_cache = linear_forward(A_prev, W, b)

	if activation == "sigmoid":
		A, activation_cache = sigmoid(Z)

	elif activation == "relu":
		A, activation_cache = relu(Z)

	assert(A.shape == (W.shape[0], A_prev.shape[1]))
	cache = (linear_cache, activation_cache)
	return A, cache

def L_model_forward(X, parameters):
	"""
	Implement forward propagation for the [LINEAR -> ReLU]*(l-1) -> LINEAR -> SIGMOID computation

	Arguments:
	X -- data, numpy array of shape(input size, number of examples)
	parameters -- output of initialize_parameters_deep()

	Returns:
	AL -- last post-activation value
	cache -- list of cache containing each layers forwards cache
	"""

	caches = []
	A = X
	L = len(parameters)// 2 #layers of neural network

	"""
	Implement [LINEAR -> ReLU] * (l-1) layers of neural network, Add cache to the list
	""" 

	for l in range(1, L):
		A_prev = A 
		A, cache = linear_activation_forward(A_prev, parameters["W"+str(l)], parameters["b"+str(l)], activation = "relu")
		caches.append(cache)

	AL, cache = linear_activation_forward(A, parameters["W"+str(L)], parameters["b"+str(L)], activation = "sigmoid")
	caches.append(cache)

	assert(AL.shape == (3, X.shape[1]))

	return AL, caches

def compute_cost(AL, Y):
	"""
	Implement the cost function

	Arguments:
	AL -- probability vector of label prediction, shape(1, number of examples)
	Y  -- ture “label” vector, shape(1, number of examples)

	Returns:
	cost -- cross-entropy cost
	"""
	m = Y.shape[1]
	cost = -np.sum(np.multiply(np.log(AL), Y) + np.multiply(np.log(1- AL),1 - Y)) / m
	
	cost = np.squeeze(cost)

	assert(cost.shape == ())

	return cost

def linear_backward(dZ,  cache):
	"""
	Implement the linear portion of backward propagation for single  layer

	Arguments:
	dZ -- Gridient of the cost with respect to the linear output 
	caches -- tuple of values （A_prev， W， b）coming form forward propagation in the current layer

	Returns:
	dA_prev -- Gridient of the cost with respect to the activation(of previous layer l-1), same shape as A_prev
	dW		-- Gridient of the cost with respect to W (current layer l)
	db		-- Gridient of the cost with respect to b (current layer l)
	"""

	A_prev, W, b = cache 
	m = A_prev.shape[1]

	dW = np.dot(dZ, A_prev.T) / m
	db = np.sum(dZ, axis = 1, keepdims = True) / m
	dA_prev = np.dot(W.T, dZ)

	assert(dA_prev.shape == A_prev.shape)
	assert(dW.shape == W.shape)
	assert(db.shape == b.shape)

	return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
	"""
	Implement the backward propagation for the LINEAR->ACTIVATION layer.

	Arguments:
	dA -- post-activation Gridient of current layer l
	cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficently
	activation -- the activation to be uesd in this layer, stored as a text string: "sigmoid" or "relu"

	Returns:
	dA_prev -- Gridient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
	dW		-- Gridient of the cost with respect to W (current layer l), same shape as W
	db		-- Gridient of the cost with respect to b (current layer l), same shape as b
	""" 

	linear_cache, activation_cache = cache

	if activation == "relu":
		dZ = relu_backward(dA, activation_cache)
		dA_prev, dW, db = linear_backward(dZ, linear_cache)

	if activation == "sigmoid":
		dZ = sigmoid_backward(dA, activation_cache)
		dA_prev, dW, db = linear_backward(dZ, linear_cache)

	return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
	"""
	Implement the backward propagation for the [LINEAR->ReLU] * (l-1) -> LINEAR -> SIGMOID group

	Arguments:
	AL -- probability vector, output of the forward propagation (L_model_forward())
	Y  -- ture "label" vector (containing[1,0,0]  if negtive, [0,1,0] if nuetral, [0,0,1] if negtive)
	caches -- list of cache containing:

	Returns:
	grads -- A dictionary with gradients
			grads["dA"+str(l)] = ...
			grads["dW"+str(l)] = ...
			grads["db"+str(l)] = ...
	"""

	grads = {}
	L = len(caches)
	m = AL.shape[1]	
	Y = Y.reshape(AL.shape)

	dAL = -(np.divide(Y, AL) - np.divide(1-Y, 1- AL))
	current_cache = caches[L-1]
	grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid")

	for l in reversed(range(L-1)):
		current_cache = caches[l]
		temp_dA_prev, temp_dW, temp_db = linear_activation_backward(grads["dA"+str(l+2)], current_cache, activation = "relu" )
		grads["dA"+str(l+1)] = temp_dA_prev
		grads["dW"+str(l+1)] = temp_dW
		grads["db"+str(l+1)] = temp_db

	return grads

def update_parameters(parameters, grads, learning_rate):
	"""
	Update parameters using gradient descent

	Arguments:
	parameters -- python dictionary for parameters
	grads -- python dictionary containing gradients 

	Returns:
	parameters -- python dictionary containing updated parameters
	"""

	L =	len(parameters)// 2
	for l in range(L):
		parameters["W"+str(l+1)] = parameters["W"+str(l+1)]-learning_rate*grads["dW"+str(l+1)]
		parameters["b"+str(l+1)] = parameters["b"+str(l+1)]-learning_rate*grads["db"+str(l+1)]

	return parameters

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=True):#lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []                         # keep track of cost
    
    parameters = initialize_parameters_deep(layers_dims)

    for i in range(0, num_iterations):


        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate=learning_rate)

        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

        if i%100 == 0 and i>100 and abs(costs[int(i/100)-1] - cost) <= 0.0001: #Avoid overfitting
            break
            
    # plot the cost
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters

def NN_predict(X, y, parameters):
    """
    This function is used to predict the results of a  L-layer neural network.
    
    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    p -- predictions for the given dataset X
    """
    
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    
    probas, caches = L_model_forward(X, parameters)

    predic = []

    
    accuracy = 0
    for i in range(0, probas.shape[1]):
        temp_prob = probas[:,i]
        index = 0
        max_prob = temp_prob[0]
        for k in range(0,len(temp_prob)):
            if temp_prob[k] > max_prob:
                max_prob = temp_prob[k]
                index = k
        if index == 0:
        	predic.append([1,0,0])
        elif index == 1:
        	predic.append([0,1,0])
        elif index == 2:
        	predic.append([0,0,1])

        if y[index, i] == 1:
            accuracy += 1

    accuracy = accuracy/m
    print("accuracy is:", accuracy)
    # print(probas[:,10])
    # print(predic[10])

    # print(y[:,10])
    return predic

def predict1(X, y, parameters):
    """
    This function is used to predict the results of a  L-layer neural network.
    
    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    p -- predictions for the given dataset X
    """
    
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1,m))
    
    # Forward propagation
    probas, caches = L_model_forward(X, parameters)

    
    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0
    
    #print results
    #print ("predictions: " + str(p))
    #print ("true labels: " + str(y))
    print("Accuracy: "  + str(np.sum((p == y)/m)))
        
    return p



