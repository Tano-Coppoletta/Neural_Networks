import numpy as np
from matplotlib import pyplot as plt

if __name__ == '__main__':
    np.random.seed(259358)
    w0= np.random.uniform(-1/4,1/4)
    w1=np.random.uniform(-1,1)
    w2=np.random.uniform(-1,1)

    n_points=1000

    S = np.random.uniform(-1, 1, size=(n_points, 3))
    #S = | x1 | x2 | classification(0/1)
    
    w = np.array([w0,w1,w2])
    S1=np.zeros((n_points,2))
    s1_cnt=0
    s2_cnt=0
    S2=np.zeros((n_points,2))
    x=np.zeros(n_points)
 
    for i in range(n_points):
        X=np.array([1, S[i][0], S[i][1]])
        #X*w
        res = np.dot(X,w)
        if(res>=0):
            S1[s1_cnt][0]=S[i][0]
            S1[s1_cnt][1]=S[i][1]
            #save the desired output
            S[i][2]=1
            s1_cnt+=1
        else:
            S2[s2_cnt][0]=S[i][0]
            S2[s2_cnt][1]=S[i][1]
            #save the desired output
            S[i][2]=0
            s2_cnt+=1
   

    plt.title("Homework 2")

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.ylim(-1.15,1.15)
    plt.grid()
    plt.scatter(S1[0:s1_cnt,0],S1[0:s1_cnt,1],color='green',marker='^',s=15)
    plt.scatter(S2[0:s2_cnt, 0], S2[0:s2_cnt, 1], color='red', marker='*', s=15)

    x = np.linspace(-1,1,1000)
    y=(-w0-w1*x)/w2
    plt.plot(x,y,'r')
    plt.legend(["x1", "x2","Boundary"])
    plt.show()

    #(h) PTA
    n=1
    w0_= random.uniform(-1,1)
    w1_=random.uniform(-1,1)
    w2_=random.uniform(-1,1)
    w=np.array([w0_,w1_,w2_])
    
    em = np.zeros((1000,2),dtype=float)
    #| epoch | misclassification | 
    index=0
    misclassification=-1
    
    while(misclassification!=0):
        #initialize the number of misclassification to 0
        misclassification=0
        em[index][1] = 0
        for i in range(0,n_points):
            X=np.array([1, S[i][0], S[i][1]])
            res = np.dot(X,w)
            if(res>=0 and S[i][2]==0):  
                #update the weights
                #weights=weights- n*xi
                
                w = w - (n*X.transpose())   
                em[index][1]+=1 
                misclassification+=1
            elif(res<0 and S[i][2]==1):
                w = w + (n*X.transpose())
                em[index][1]+=1
                misclassification+=1
        print("Misclassification ",em[index][1],"Epoch: ",em[index][0])
        em[index+1][0]=em[index][0]+1
        index+=1

    plt.title("Epoch vs n. of misclassificatoins")
    plt.xlabel("Epoch")
    plt.ylabel("Number of misclassifications")
    plt.plot(em[0:index,0],em[0:index,1],'r')
    plt.show()

    plt.title("Homework 2")

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.ylim(-1.15,1.15)
    plt.grid()
    plt.scatter(S1[0:s1_cnt,0],S1[0:s1_cnt,1],color='green',marker='^',s=15)
    plt.scatter(S2[0:s2_cnt, 0], S2[0:s2_cnt, 1], color='red', marker='*', s=15)
    x = np.linspace(-1,1,100)
    y=(-w[0]-w[1]*x)/w[2]
    plt.plot(x,y,'r')
    plt.legend(["x1", "x2","Boundary"])
    plt.show()
        


