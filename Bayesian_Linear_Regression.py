import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
def dataset(N):
    x,y = [],[]
    for i in range(N):
        xx = float(i)/float(N-1)
        x.append(xx)
        y.append(f(xx)+ np.random.normal(scale=0.5))
    return x,y
def f(x):
    return np.cos(2.0*np.pi*x)*np.tan(x)#np.sin(2.0*np.pi*x)

class Byse_fitting():
    def __init__(self,alpha,beta,degree):
        self.alpha = alpha
        self.beta = beta
        self.degree = degree

    def fit(self,x,y):
        x,y = np.array(x),np.array(y)
        self.y = y
        n = x.shape[0]
        phi = np.zeros([n,self.degree+1])
        I = np.eye(self.degree+1)
        for i in range(self.degree+1):
            phi[:,i] = x ** i
        self.phi=phi
        S_inv = self.beta*np.dot(phi.T,phi) + self.alpha*I
        self.S = np.linalg.inv(S_inv)

        
    def _phi(self,xx,M):
        return np.array([xx ** i for i in range(M+1)]).reshape(self.degree+1,1)
                        
    def s(self,xx):
        new_phi = self._phi(xx,self.degree)
        return np.diag(np.sqrt(1./self.beta + np.dot(np.dot(new_phi.T,self.S),new_phi)))
    
    
    def m(self,xx):    
        new_phi =  self._phi(xx,self.degree)
        
        return self.beta * np.dot(self.S,np.dot(self.phi.T,self.y))
    
    def prediction_distribute(self,x,y):
        new_phi = self._phi(x,self.degree)
        mu_ML =np.dot(new_phi.T,self.m(x))
        S_ML = self.s(x)
        normal_distribute = (1/np.sqrt(2*np.pi*S_ML))*np.exp(-((y-mu_ML)**2)/(2*S_ML))
        return normal_distribute

# 結果の表示


beta = 1./(0.2)**2
alpha = 0.01
degree = 10
xx = np.linspace(0,1.01,200)
fig = plt.figure(figsize=(10,10))
x_p, y_p = np.meshgrid(np.linspace(-2, 2, 100), np.linspace(-4.0, 4.0, 100))

for i in range(12):
    x,y = dataset(25)
    model = Byse_fitting(alpha,beta,i)
    model.fit(x,y)
    
    ax = fig.add_subplot(4,3,i+1)
    ax.set_xlim([-0.25,1.25])
    ax.set_ylim([-4,4])
    ax.plot(xx,f(xx),label = "True",c="b")
    #mean = model.m(xx)
    #sigma = model.s(xx)
    
    vec_normal = np.vectorize(model.prediction_distribute)
    z = vec_normal(x_p,y_p)
    ax.contourf(x_p, y_p, z,100,cmap=cm.PuBu_r)
    ax.scatter(x,y,c = "k",s=30)
   
    

