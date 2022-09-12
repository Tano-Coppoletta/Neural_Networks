import random
import numpy as np

if __name__ == '__main__':

    seed= random.uniform(-1/4,1/4)
    w1=random.uniform(-1,1)
    w2=random.uniform(-1,1)
    print(seed ," ",w1," ",w2)

    n=100

    data = np.random.uniform(-1, 1, size=(6, 2))

