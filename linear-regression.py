import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# 1) Design model (input, output size, forward pass)

# 2) Construct loss and optimizer

# 3) Training loop

# forward pass: compute prediction and loss

# - backward pass: gradients
# - update weights


# 0) DATA PREPARATION

# generates 100 samples where each data point has 1 feature
# the noise controls the amount of variability
# random_state parameter sets the seed for the random number generator used to generate the data
# the random_state parameter is set to 4, which means that the same synthetic regression dataset 
# will be generated each time this code is run with random_state=4
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)

# converts the data to a float32 Tensor
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))
# we want to reshape the y because its currently one row
# we want to reshape it to a column vector so we want to put each value in its own row
# .view() is a method that reshapes a tensor
y = y.view(y.shape[0], 1)

# get number of samples and number of features
n_samples, n_features = X.shape

# 1) MODEL

# Linear model f = wx + b
# input size is the number of features, which is 1 - set previously
input_size = n_features
# output size is 1 because we want to have one output value for every input value
output_size = 1
# model gets input and output size
model = nn.Linear(input_size, output_size)

# 2) LOSS AND OPTIMIZER

learning_rate = 0.01

# with pytorch, this initializes an instance of the Mean Squared Error (MSE) loss function,
# which is used as a criterion to measure the difference between the predicted output
# and the target output in a regression problem.
criterion = nn.MSELoss()

# initializes an instance of the Stochastic Gradient Descent (SGD) algorithm/optimizer with a given learning rate
# this implements the stochastic gradient descent algorithm
# The model.parameters() method is used to get the parameters of the PyTorch model that need to be optimized. 
# These parameters include the weights and biases of the neural network layers. 
# The lr parameter specifies the learning rate, which determines the step size 
# of the optimizer during parameter updates and is defined above as .01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

# 3) TRAINING LOOP: FORWARD PASS, LOSS, BACKWARD PASS, UPDATE WEIGHTS
# Keep in mind that during the forward pass of the model, the input data is fed into the model and 
# the output is computed. The backward pass then works backwards from the output of the model to 
# calculate how much each parameter of the model contributed to the error (loss) during training.

# number of epochs refers to the number of times a machine learning algorithm, such as neural network, 
# is trained on the entire training dataset during the training process. In this case, it's 100
num_epochs = 100
# creates a for loop that iterates over a range of num_epochs, where num_epochs is an integer variable 
# representing the number of epochs to train a machine learning model.
for epoch in range(num_epochs):

    # Forward pass
    # runs the machine learning model on the input data (X) to obtain a predicted output (y_predicted)
    y_predicted = model(X)
    # then calculates the difference between the predicted output and the true output (y) 
    # using a pre-defined loss function (criterion)
    loss = criterion(y_predicted, y)
    
    # Backward pass
    # computes the gradient of the loss function with respect to the model parameters
    loss.backward()
    # optimizer determines how much to adjust the parameters based on the gradients of the loss function
    # and updates the weights
    optimizer.step()

    # after each optimization step, the gradients of the previous step are still stored in the memory, 
    # and if they are not cleared, they will be added to the new gradients in the next step. 
    # This can cause the model to update its parameters based on a combination of old and new gradients, 
    # which may lead to unexpected behavior and suboptimal performance.
    # So, use zero grad before new step in order to update/empty gradients
    optimizer.zero_grad()

    # print loss every 10 epochs
    # f'epoch: {epoch+1}, loss = {loss.item():.4f}' syntax is a formatted string that combines 
    # the epoch number and the loss value into a single string.
    if (epoch+1) % 10 == 0:
        # loss.item():.4f part of the string formats the loss value as a float with 4 decimal places.
        # simple way to provide the user with some information about the progress 
        # of the training process during the execution of the program.
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

# Plot
# model(X) code generates the predictions on the input data using the trained PyTorch model
# .detach() method is used to detach the tensor from the computational graph, 
# which means that the tensor is no longer part of the graph and no gradient will be backpropagated through it
# .numpy() method is then called on the detached tensor to convert it into a NumPy array
predicted = model(X).detach().numpy()

# plots the input data as red circles ('ro') on the plot. The X_numpy and y_numpy arrays contain the input 
# features and output values, respectively, and the 'ro' format string specifies that the markers should be red circles.
plt.plot(X_numpy, y_numpy, 'ro')
# plt.plot(X_numpy, predicted, 'b'), plots the predicted output of the model as a blue line ('b') on the same plot. 
# The X_numpy array contains the input features, and predicted contains the output values predicted by the model.
plt.plot(X_numpy, predicted, 'b')
# plt.show() method is called to display the plot on the screen.
plt.show()