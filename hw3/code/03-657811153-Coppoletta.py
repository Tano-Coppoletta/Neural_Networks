import numpy as np
import matplotlib.pyplot as plt
import gzip
import random

if __name__ == '__main__':
    
    f = gzip.open('../data/train-images-idx3-ubyte.gz','r')

    image_size = 28

    #decide n <=6000
    n = 5

    f.read(16)
    buf = f.read(image_size * image_size * n)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(n, image_size, image_size, 1)
    print(data[0][0])

    image = np.asarray(data[3]).squeeze()
    plt.imshow(image)
    plt.show()

    f = gzip.open('../data/train-labels-idx1-ubyte.gz','r')
    f.read(8)
    for i in range(0,n):   
        buf = f.read(1)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
       # print(labels)

    
    
    #parameters
    eta=1   
    epsilon=0

    #initialize random weights
    W = np.random.uniform(-1, 1, size=(784, 10))
    epoch=0
    #here i am assuming a maximum of 1000 epochs
    errors = np.zeros((1000,1),dtype=float)

    while(errors[epoch-1]/n > epsilon):
        for i in range(0,n):
            v=np.dot(W,data[i]) #v=Wxi
            j=np.argmax(v)
            if(j!=labels[i]):
                errors[epoch]=errors[epoch]+1
        epoch=epoch+1
        for z in range(0,n):
            W=W+eta*np.dot((labels[z]-np.heaviside(np.dot(W,data[z]),1)),data[z].transpose())


