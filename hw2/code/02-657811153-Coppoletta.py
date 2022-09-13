import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':

    w0= np.random.uniform(-1/4,1/4)
    w1=np.random.uniform(-1,1)
    w2=np.random.uniform(-1,1)
    print(w0 ," ",w1," ",w2)

    n=100

    S = np.random.uniform(-1, 1, size=(100, 2))
    print(S)
    print("\n\n")
    #build S1
    w = np.matrix([[w0],[w1],[w2]])
    print(w)
    S1=np.zeros((100,2))
    s1_cnt=0
    s2_cnt=0
    S2=np.zeros((100,2))
    x=np.zeros(100)
 #   print(S)
    for i in range(n):
        X=[1, S[i][0], S[i][1]]
      #  x.append([1, S[i][0], S[i][0]])
       # print(x)
        res = np.dot(X,w)
       # print(res)
        if(res>=0):
            S1[s1_cnt][0]=S[i][0]
            S1[s1_cnt][1]=S[i][1]
            s1_cnt+=1
        else:
            S2[s2_cnt][0]=S[i][0]
            S2[s2_cnt][1]=S[i][1]
            s2_cnt+=1
    print(S1)
    print(S2)

    plt.title("Homework 2")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.plot(x[1],x[2])
    plt.scatter(S1[0:s1_cnt,0],S1[0:s1_cnt,1],color='green',marker='^',s=15)
    plt.scatter(S2[0:s2_cnt, 0], S2[0:s2_cnt, 1], color='red', marker='*', s=15)

    plt.show()



