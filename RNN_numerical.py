#Implement Vanilla RNN from scratch

import numpy as np
import pandas as pd

X1=np.array([
    [1],[2]
          ])
X2=np.array([
    [-1],[1]
           ])
X=[X1,X2]#store the inputs

h0=np.array([[0],[0],[0]])
h=h0

Wxh=np.array([[0.5,-0.3],
              [0.8,0.2],
              [0.1,0.4]])
Whh=np.array([[0.1,0.4,0],
              [-0.2,0.3,0.2],
              [0.05,-0.1,0.2]])
Why=np.array([[1,-1,0.5],
              [0.5,0.5,-0.5]])
for x in X:

  ht=np.tanh(Whh @ h + Wxh @ X )
  print("hidden state:",ht)

  y1=Why @ ht
  y1=1/(1+np.exp(-y1))
  print("Output:",y1)