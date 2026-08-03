#Implement the following functions in Python from scratch. Do not use any library functions.
# You are allowed to use numpy and matplotlib. Generate 100 equally spaced values between -10 and 10.
# Call this list as  z. Implement the following functions and its derivative.
# Use class notes to find the expression for these functions.
# Use z as input and plot both the function outputs and its derivative outputs.
# Upload your code into Github and share it with me.
#Sigmoid
#Tanh
#ReLU (Rectified Linear Unit)
#Leaky ReLU
#Softmax (no need for visualization)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


"""generating 100 equally spaced points from -10 to 10"""
z=np.linspace(-10,10,100)
print(z)

"""------------sigmoid function and its derivative function--------------"""
def sigmoid(z):
    return 1/(1+np.exp(-z))
s_z=sigmoid(z)
def sigmoid_plot(z,s_z):
    plt.figure(figsize=[5,5])
    plt.plot(z,s_z)
    plt.xlabel('z')
    plt.ylabel('s_z')
    plt.title('sigmoid function')
    plt.legend()
    plt.grid(True)
    p_sd=plt.show()
    return p_sd
sigmoid_plot(z,s_z)
print("The sigmoid points are:",s_z)


def sigmoid_derivative(z):
    sigmoid=1/(1+np.exp(-z))
    return sigmoid*(1-sigmoid)
sd_z=sigmoid_derivative(z)
print("The sigmoid derivative points are:",sd_z)

def plot_sigmoid_d(z,sd_z):
  plt.figure(figsize=(5,5))
  plt.plot(z,sd_z)
  plt.xlabel("z")
  plt.ylabel("sigmoid derivative")
  plt.title("sigmoid derivative function")
  plt.legend()
  plt.grid(True)
  p_sig=plt.show()
  return p_sig
plot_sigmoid_d(z,sd_z)


"""-------------tanh function and its derivative function--------------"""
def tanh(z):
    return np.exp(z)-np.exp(-z)/np.exp(z)+np.exp(-z)
t_z=tanh(z)
print("The tanh points are:",t_z)


def tanh_derivative(z):
    exp_cal=((np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z)))**2
    return 1-exp_cal
td_z=tanh_derivative(z)
print("The tanh derivative points are:",td_z)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(z,np.tanh(z))
plt.xlabel('z')
plt.ylabel('tanh(z)')
plt.title('tanh function')
plt.subplot(1,2,2)
plt.plot(z,np.tanh(td_z))
plt.title('tanh derivative function')
plt.xlabel('z')
plt.ylabel('tanh(z) derivative')
plt.grid(True)
plt.show()

"""-------------relu function and its derivative function--------------"""
def relu(z):
 relu=[]
 for values in z:
    if values >0:
        relu.append(values)
    else:
        relu.append(0)
 return relu
relu_1=relu(z)
print("The relu points are:",relu_1)

def relu_derivative(z):
  relu_derivative=[]
  for values in z:
      if values >0:
          relu_derivative.append(1.0)
      else:
          relu_derivative.append(0.0)
  return relu_derivative
relu_d=relu_derivative(z)
print("The relu derivative points are:",relu_d)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(z,relu_1)
plt.xlabel('z')
plt.ylabel('relu(z)')
plt.title('ReLU function')
plt.grid(True)
plt.subplot(1,2,2)
plt.plot(z,relu_d)
plt.xlabel('z')
plt.ylabel('relu(z) derivative')
plt.title('ReLU derivative')
plt.grid(True)
plt.tight_layout()
plt.show()

"""-------------Leaky ReLU function and its derivative function--------------"""
def l_relu(z):
    Leaky_relu=[]
    alpha=0.01
    for values in z:
        if values >0:
            Leaky_relu.append(values)
        else:
            Leaky_relu.append(alpha*values)
    return Leaky_relu
leaky_relu_1=l_relu(z)
print("The Leaky ReLU points are:",leaky_relu_1)

def leaky_relu_derivative(z):
    leaky_relu_derivative=[]
    alpha=0.01
    for values in z:
        if values >0:
            leaky_relu_derivative.append(1.0)
        else:
            leaky_relu_derivative.append(alpha*values)
    return leaky_relu_derivative
lrd=leaky_relu_derivative(z)
print("The Leaky ReLU derivative points are:",lrd)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(z,leaky_relu_1)
plt.xlabel('z')
plt.ylabel('leaky_relu_1')
plt.title('Leaky ReLU function')
plt.grid(True)
plt.subplot(1,2,2)
plt.plot(z,lrd)
plt.xlabel('z')
plt.ylabel('lrd')
plt.title("Leaky ReLU derivative")
plt.grid(True)
#plt.show()

"""---------------------Softmax----------------------------------"""
def softmax(z):
    exp_cal = np.exp(z)
    softmax_values = exp_cal / np.sum(exp_cal)
    return softmax_values
softmax_1 = softmax(z)
print("Softmax points are:", softmax_1)
