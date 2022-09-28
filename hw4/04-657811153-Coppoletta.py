from ast import Num
import numpy as np
import matplotlib.pyplot as plt
import gzip
import random

def identity_function(x):
    return x

def identity_derivative(x):
    return 1

def mean_squared_error(d,f):
    return ((d-f)**2).sum() / d.size

if __name__ == '__main__':
    np.random.seed(1)
    n = 300
    x = np.random.uniform(0,1,(n))
    v = np.random.uniform(-1/10,1/10,(n))

    d = np.sin(20*x)+3*x+v

    plt.title("Plot (xi,di)")

    plt.xlabel("xi")
    plt.ylabel("di")
  #  plt.ylim(-1.15,1.15)
    plt.grid()
    plt.scatter(x,d,color='green',marker='^',s=15)
    plt.show()

    input_size=1
    hidden_layer=24
    output_size=1

    

    W1 = np.random.uniform(-1,1,(input_size,hidden_layer))
    W2 = np.random.uniform(-1,1,(hidden_layer,output_size))
    #25 bias, the first 24 are for the first layer
    bias = np.random.uniform(-1,1,(1,hidden_layer+output_size))

    for i in range(n):
        #feedforward 
        A=np.dot(x[i],W1)
        Ab=np.sum(A,bias[:hidden_layer])
        A_activated= np.tanh(Ab)

        B=np.dot(A_activated,W2)
        Bb=np.sum(B,bias[-1])
        output= identity_function(Bb)

        #feedback
        #derivative of tanh = 1-tanh^2
        O = d[i] - output
        C = identity_derivative(O)

        xw = np.dot(x[i],W1) 
        xwb = xw + bias[:hidden_layer]
        D = 1- np.tanh(xwb)**2 

       # W1 = W1 - eta * 


    eta=1

