import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

# 1) Design model (input, output size, forward pass)

# 2) Construct loss and optimizer

# 3) Training loop
#   - forward pass: compute prediction and loss
#   - backward pass: gradients
#   - update weights

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 0) Prepare data
bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target

# gets the numbers of samples and features
n_samples, n_features = X.shape

# The X and y variables contain the input features and target variable, respectively, of the dataset.
# test_size parameter specifies the proportion of the dataset that should be used for testing. 
# In this case, it is set to 0.2, which means that 20% of the data will be used for testing and 80% for training.
# setting random_state to a specific integer value (in this case, 1234), the same random sequence will be generated 
# every time the code is run, which ensures that the same train-test split is obtained each time the code is executed.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# scale
# using the fit() method, the model learns the scaling or transformation parameters based on the training data.
# Then, when you call the transform() method on the same training dataset, 
# it applies the learned transformation to the training data.
# transform() method is a way to apply a pre-processing step to new data or a specific set of features, 
# using the same pre-processing parameters that were learned from the training data.
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# convert in numpy arrays into PyTorch tensors
# First, the astype() method is used to convert the NumPy arrays to np.float32
# Then, the torch.from_numpy() method is used to convert the NumPy arrays to PyTorch tensors. 
# By converting the data to PyTorch tensors, the data can be used as inputs to PyTorch deep 
# learning models for training and inference. 
X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

# puts each value in 1 row with 1 column
# expected format for many PyTorch deep learning models, which can simplify the code when building and training models.
y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)

# 1) Model
# Linear model f = wx + b , sigmoid at the end 
class Model(nn.Module):
    #  __init__() method initializes the model by creating a single linear layer using the nn.Linear() class. 
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        # The linear layer takes n_input_features as its input dimension and 1 as its output dimension.
        self.linear = nn.Linear(n_input_features, 1)
    # forward() method defines the forward pass of the model. It takes an input tensor x and passes it through the linear layer, 
    # followed by a sigmoid activation function using the torch.sigmoid() method. The output of the model is the predicted output y_pred.
    def forward(self, x):
        # The sigmoid activation function is often used for binary classification problems, as it maps the output of the linear layer 
        # to a value between 0 and 1, which can be interpreted as a probability.
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred
# creates an instance of the Model class and assigns it to the variable model, 
# with n_features as the input dimension of the linear layer.
model = Model(n_features)

# 2) Loss and optimizer
# number of times the entire training dataset will be used to update the model during training.
num_epochs = 100
# step size used by the optimizer during training. This value can be adjusted to control how quickly or 
# slowly the optimizer updates the model parameters.
learning_rate = 0.01
# nn.BCELoss() is used to define the loss function for the model. BCE stands for Binary Cross Entropy, 
# and is commonly used for binary classification problems where the model predicts a binary output (0 or 1).
criterion = nn.BCELoss()
# used to define the optimizer for the model. In this case, the optimizer is Stochastic Gradient Descent (SGD)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 3) Training loop
for epoch in range(num_epochs):
    # Forward pass and loss
    # takes the X_train input data as input and makes a forward pass to generate predicted outputs y_pred.
    y_pred = model(X_train)
    # loss function (criterion) is used to calculate the difference between the predicted 
    # outputs y_pred and the actual target values y_train.
    loss = criterion(y_pred, y_train)

    # Backward pass and update
    # backward() method is called on the loss to compute the gradients of the model parameters with respect to the loss.
    loss.backward()
    # optimizer's step() method is called to update the model parameters using the gradients computed in step 3.
    optimizer.step()

    # zero grad before new step
    # optimizer's zero_grad() method is called to reset the gradients to zero before the next iteration.
    optimizer.zero_grad()

    # Every 10 epochs, the loss is printed to monitor the progress of the training.
    if (epoch+1) % 10 == 0:
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')

# computes the accuracy of the PyTorch model on a test dataset using the with torch.no_grad() context manager. 
# This context manager temporarily sets requires_grad attribute to False for all tensors inside the block, 
# which means that PyTorch will not track gradients for those tensors. This is useful when we want to compute
# the forward pass of the model without updating its parameters or computing gradients.
with torch.no_grad():
    # y_predicted is calculated by passing the X_test input data to the trained model.
    # y_predicted_cls is calculated by rounding the y_predicted tensor, as the model 
    # output is a continuous value but the target variable is binary (0 or 1).
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    # accuracy of the model is calculated by comparing the predicted binary values in y_predicted_cls 
    # with the actual binary values in y_test. The eq() method compares the two tensors element-wise 
    # and returns a tensor of the same shape containing True or False values. The sum() method is then 
    # used to count the number of correct predictions, and the float() method is used to convert the result to a float. 
    # Finally, the accuracy is calculated as the fraction of correct predictions out of the total number of predictions.
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy: {acc.item():.4f}')