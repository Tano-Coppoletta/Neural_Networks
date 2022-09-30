import numpy as np
import matplotlib.pyplot as plt

def identity_function(x):
    return x

def mean_squared_error(d,f):
    return np.sum((d-f)**2)/300 

def tanh_derivative(x):
    return 1- np.tanh(x)**2
    
if __name__ == '__main__':
    np.random.seed(234)
    n = 300
    x = np.random.uniform(0,1,(n))
    v = np.random.uniform(-1/10,1/10,(n))
    N=24
    d = np.sin(20*x)+3*x+v

    plt.title("Plot (xi,di)")

    plt.xlabel("xi")
    plt.ylabel("di")
    plt.grid()
    plt.scatter(x,d,color='green',marker='^',s=15)
    plt.show()

    eta=0.01

    

    W1 = np.random.randn(N,2) # weights + bias vector
    #col 0 = bias
    #col 1 = weights
    W2 = np.random.randn(1,N+1) # last bias + weights
    # | b | w0 | w1 | ... |w_N-1 |

    epochs=10000
    epoch_vs_mse = np.zeros((2,epochs))
    outputs = np.zeros(n)
    index=0
    epoch=0

    while(epoch!=epochs):
        epoch_vs_mse[0,epoch]=epoch
        for i in range(n):
            #induced local field 1
            vl1 = W1 @ [[1],[x[i]]] 
            y1 = np.tanh(vl1)   
            #induced local field 2
            y2 = np.append([1],y1)
            vl2 = np.dot(W2,y2)   
            outputs[index]=identity_function(vl2)
           
            #backpropagation
            deltaL = d[i]-outputs[index]
            delta1 = np.multiply((np.dot(np.transpose(W2), deltaL.reshape(1,1)))[1:,:],tanh_derivative(vl1))
            index+=1

            #update the weights
            W1 = W1 + (eta * np.dot(delta1, [[1,x[i]]]))
            W2 = W2 + (eta* np.dot(deltaL, y2))

        
        epoch_vs_mse[1,epoch] = mean_squared_error(d,outputs)
       # if(epoch != 0 and epoch_vs_mse[1,epoch]>epoch_vs_mse[1,epoch-1]):
       #     eta=0.9*eta
        print(epoch,epoch_vs_mse[1,epoch])
        if(epoch_vs_mse[1,epoch]<0.01):
            print("End at epoch",epoch)
            break
        epoch+=1
        index=0
    #plot
    plt.title("Epoch vs MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.plot(epoch_vs_mse[0,1:epoch],epoch_vs_mse[1,1:epoch],'r')
    plt.show()

    xs, ys = zip(*sorted(zip(x, outputs)))
    plt.xlabel("xi")
    plt.ylabel("di")
    plt.grid()
    plt.scatter(x,d,color='green',marker='^',s=15)
    plt.plot(xs,ys,'r')
    plt.show()




