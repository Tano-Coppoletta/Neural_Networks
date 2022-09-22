import numpy as np
import matplotlib.pyplot as plt
import gzip
import random

def stepFunction(input):
    output = np.zeros((len(input),1),dtype=float)
    for i in range(0,len(input)):
        if(input[i]>=0):
            output[i]=1
        else:
            output[i]=0
    return output


if __name__ == '__main__':
    np.random.seed(7)
    f = gzip.open('../data/train-images-idx3-ubyte.gz','r')

    image_size = 28

    #decide n <=6000
    n = 1000

    f.read(16)
    buf = f.read(image_size * image_size * n)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(n, image_size, image_size, 1)
   # print(data[0][0])
    images=np.zeros((784,n),dtype=float)
    for i in range(0,n):
        images[:,i]= np.asarray(data[i]).reshape(-1)

    f = gzip.open('../data/train-labels-idx1-ubyte.gz','r')
    f.read(8)
    labels=np.zeros(n)
    for i in range(0,n):   
        buf = f.read(1)
        labels[i]=np.frombuffer(buf, dtype=np.uint8)
   
    
    #parameters
    eta=1   
    epsilon=0

    #initialize random weights
    W = np.random.uniform(-1,1,(10, 784))
    epoch=0
   # err_1=0
    em = np.zeros((99999,2),dtype=float)
    #| epoch | misclassification | 
   # errors = np.zeros((99999999,1),dtype=float)
   # em[0,0]=1
    for i in range(0,n):
        v=np.dot(W, images[:,i]) #v=Wxi
        j=np.argmax(v)
        if(j!=labels[i]):
            em[epoch,1]=em[epoch,1]+1
        
        
    for z in range(0,n):
        label_bin=np.zeros((10,1))
        label_bin[int(labels[z])]=1
        Wxi=np.dot(W,images[:,z]) #Wx_i
        W=W+eta*(np.dot((label_bin-stepFunction(Wxi)),(images[:,z]).reshape(1,784)))


    epoch=epoch+1
    while(em[epoch-1,1]/n > epsilon):
        em[epoch,0]=epoch
        for i in range(0,n):
            v=np.dot(W, images[:,i]) #v=Wxi
            j=np.argmax(v)
            if(j!=labels[i]):
                em[epoch,1]=em[epoch,1]+1
        epoch=epoch+1
        
        for z in range(0,n):
            label_bin=np.zeros((10,1))
            label_bin[int(labels[z])]=1
            Wxi=np.dot(W,images[:,z]) #Wx_i
            #data_t=data[z]
            W=W+eta*(np.dot((label_bin-stepFunction(Wxi)),(images[:,z]).reshape(1,784)))
            
   # print(errors)

    plt.title("Epoch vs n. of misclassifications")
    plt.xlabel("Epoch")
    plt.ylabel("Number of misclassifications")
    plt.plot(em[0:epoch,0],em[0:epoch,1],'r')
    plt.show()

    #testing 
    num_test_images=10000
    f = gzip.open('../data/t10k-images-idx3-ubyte.gz','r')
    f.read(16)
    buf = f.read(image_size * image_size * num_test_images)
    test_data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    test_data = test_data.reshape(num_test_images, image_size, image_size, 1)
    test_images = np.zeros((784,num_test_images),dtype=float)
    for i in range(0,num_test_images):
        test_images[:,i]=np.asarray(test_data[i]).reshape(-1)

    f = gzip.open('../data/t10k-labels-idx1-ubyte.gz','r')
    f.read(8)
    test_labels=np.zeros(num_test_images)
    for i in range(0,num_test_images):   
        buf = f.read(1)
        test_labels[i] = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
   
    test_errors=0
    
    for i in range(0,num_test_images):
            v_prime=np.dot(W,test_images[:,i]) #v'=Wx'_i
            j=np.argmax(v_prime)
            if(j!=test_labels[i]):
                test_errors=test_errors+1
    print("Test errors",test_errors)
