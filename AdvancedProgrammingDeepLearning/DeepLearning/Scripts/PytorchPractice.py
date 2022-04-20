############################### Lib ###############################
import torch as tc
import numpy as np
import matplotlib.pyplot as plt
####################################################################

# RVS
x = tc.tensor([ i for i in range(10)], dtype=tc.float32)
# Weights
w = tc.tensor(0. , requires_grad=True ,dtype=tc.float32)
# True model
noise = tc.normal(size=(100,1) , mean= 0  , std= 1 , dtype=tc.float32)


def true_model(x):
 return  abs(np.sin(x))+3

y_true = true_model(x)

# Forward propagation to get y_hat
def forward(x):
 return w*x

# Loss function
def loss(y_true, y_hat):
 return  ( (y_hat - y_true)**2 ).mean()


# Training
learning_rate = 0.0001
n_iters = 2500
for epoch in range(n_iters):
  # Get prediction
  y_hat = forward(x)
  # Calculate loss
  L = loss(y_true, y_hat)

  # Backprop
  L.backward()

  #exclude from computational graph in pytorch
  with tc.no_grad():
   w -=  learning_rate * w.grad


  # Zero grad to dont accumulate in torch
  w.grad.zero_()

  #verbose every multiple of 10
  if epoch % 10 ==0:
   print(f'Current Iteration: {epoch}, current weight: {w:.3f} , current loss: {L:.5f}')

print('')
print(f'Prediction after training: f(5) = {forward(5).item():.3f}')
print(f'True value f(5) : {true_model(5):.3f}')
# We dont need to compute analytical expression for the gradient