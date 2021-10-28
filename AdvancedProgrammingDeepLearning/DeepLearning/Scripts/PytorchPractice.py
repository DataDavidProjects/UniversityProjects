
############################### Lib ###############################
import torch as tc
import numpy
import matplotlib.pyplot as plt
####################################################################

####################################################################
#Define X as a tensor of linear spaced elements
x  = tc.linspace(0,21,20, requires_grad=True)

#Compute y as a function sin(x) of tensor x
y = tc.sum(tc.sin(x))

#Backprop
y.backward()
x.grad
####################################################################


# Proof that d[f(x)=sin(x)]/dx from backprop torch is equal to theoretical d[f(x)=sin(x)]/dx  = cos(x)
####################################################################
# Plotting x , dx
plt.plot(
    x.detach().numpy(),
    x.grad.detach().numpy(),
    label='Torch d[f(x)=sin(x)]/dx', linewidth = 3, linestyle = 'dashdot', color = 'navy'
)
#Plotting x , cosx  to prove cosx = d[f(x) = sin(x)]/dx
plt.plot(
    x.detach().numpy(),
    tc.cos(x).detach().numpy(),
    color = 'darkred',label='Cos(x)',alpha = 0.8, linestyle = '--')

plt.title("Function comparison")
plt.xlabel("X")
plt.legend(loc ='upper right')
plt.show()
####################################################################