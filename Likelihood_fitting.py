import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

def f(x):
    return np.sin(2*np.pi*x)

def dataset(N):
    x,y = [],[]
    for i in range(1,N):
        xx = float(i)/float(N-1)
        x.append(xx)
        y.append(f(xx)+ np.random.normal(scale=0.2))
    return x,y

class Likelehood_fitting():
    def __init__(self,degree):
        self.degree = degree
        self.beta_=0.
        self.sigma = 0.
        self.w_ = None
    
    def func(self,x):
        r=0
        for i in range(self.degree + 1):
            r += x**i * self.w_[i]
        return r
    
    def fit(self,x,y):
        x,y = np.array(x),np.array(y)
        n = x.shape[0]
        phi = np.zeros([n,self.degree+1])
        for i in range(0,self.degree+1):
            phi[:,i] =  x**i
        
        A = np.dot(phi.T,phi)
        b = np.dot(phi.T,y)
        self.w_ = linalg.solve(A,b)
        #self.w_ = np.dot(np.linalg.inv(A),b)
        E = self.E(x,y)
        self.beta_ = n/E
        self.sigma = np.sqrt(1/self.beta_)
        
    def E(self,x,y):
        E = 0.
        for x_,t_ in zip(x,y):
            E += (self.func(x_) - t_)**2
        return E
    def log_likelibood(self,x,y):
        n = len(x)
        E = self.E(x,y)
        beta = n/E
        Log_p = 0.5*(-beta*E + n*np.log(beta/(2*np.pi)))
        return Log_p

    
trainx,trainy = dataset(10)
testx,testy = dataset(10)

fig = plt.figure(figsize=(15,9))

x = np.arange(0,1,0.01)
for i,d in enumerate([1,2,3,9]):
    model = Likelehood_fitting(d)
    model.fit(trainx,trainy)
    
    ax = fig.add_subplot(2,2,i+1)
    ax.set_xlim([-0.1,1.1])
    ax.set_ylim([-1.5,1.5])
    ax.scatter(trainx,trainy,c = "k",s=50)
    ax.plot(x,f(x),label = "True",c="b")
    ax.plot(x,model.func(x),label = f"pred, beta{round(model.sigma,1)}",c = "r")
    ax.fill_between(x, model.func(x)+model.sigma, model.func(x)-model.sigma,
                     color='c', alpha=0.5, label="predict_std")
   
    ax.legend(loc=1)
    
fig.show()
fig = plt.figure()
trlp,telp=[],[]
for i in range(0,10):
    model = Likelehood_fitting(i)
    model.fit(trainx,trainy)
    trlp.append(model.log_likelibood(trainx,trainy))
    telp.append(model.log_likelibood(testx,testy))
plt.plot(np.arange(0,10),trlp,label="train_likelhood")
plt.plot(np.arange(0,10),telp,label="test_likelhood")
plt.show()
