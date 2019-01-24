import numpy as np
import matplotlib.pyplot as plt

class Bayesian_Regression():
    def __init__(self,alpha,beta):
        self.alpha = alpha
        self.beta = beta
    
    def fit(self,phi,t):
        I = np.eye(phi.shape[1])
        self.S_inv = self.alpha * I + self.beta * phi.T.dot(phi)
        self.S =  np.linalg.inv(self.S_inv)
        self.m_N = self.beta * self.S.dot(phi.T.dot(t))
        
    def predict(self,phi):
        pred_mean = phi.dot(self.m_N)
        pred_sigma =np.sqrt(1/self.beta+ np.diag(phi.dot(self.S).dot(phi.T)))
        return pred_mean,pred_sigma
    
class Evidence_Approximation(Bayesian_Regression):
    def __init__(self,alpha,beta,iter_=100):
        self.alpha = alpha
        self.beta = beta
        self.iter=iter_
    def fit(self,phi_,t):
        phi2 = phi_.T.dot(phi_)
        eig = np.linalg.eigvalsh(phi2)
        lambda_ = eig*self.beta
        for i in range(self.iter):
            old_alpha ,old_beta = self.alpha,self.beta
            super().fit(phi_,t)
            self.gamma = np.sum(eig*self.beta/(self.alpha + eig*self.beta))
            self.alpha = self.gamma/self.m_N.dot(self.m_N)
            #self.alpha = phi_.shape[1] / self.m_N.dot(self.m_N) when N >> M
            self.beta = (len(t) - self.gamma)/np.sum((t - phi_.dot(self.m_N))**2)
            #self.beta = phi_.shape[0]/np.linalg.norm((t - phi_.dot(self.m_N))**2) when N >> M
            if (abs(old_alpha-self.alpha) and abs(old_beta-self.beta))<=10e-10:
                break
        super().fit(phi_,t)
        print(f"gamma = {self.gamma}  alpha = {self.alpha}  beta = {self.beta}")
        
    def evidence_function(self,phi_,t):
        phi2 = phi_.T.dot(phi_)
        N,M = phi_.shape
        return 0.5*(M*np.log(self.alpha) + N*np.log(self.beta)
                   -self.beta * np.linalg.norm((t-phi_.dot(self.m_N))**2)-self.alpha*self.m_N.dot(self.m_N)
                    -np.log(np.linalg.det(self.S_inv)) - N*np.log(np.pi*2))

    
def dataset(N):
    x,y = [],[]
    for i in range(N):
        xx = float(i)/float(N-1)
        x.append(xx)
        y.append(func(xx)+ np.random.normal(scale=0.1))
    return np.array(x),np.array(y)

def func(x):
    return np.cos(2*np.pi*x)*np.arccos(x)

def poly(x,degree):
    return np.array([x ** i for i in range(degree+1)]).T

degree = 10


fig = plt.figure(figsize=(15,15))
xx = np.linspace(0,1.01, 300)
x,y = dataset(20)
evidence_list = []

for i in range(12):

    X = poly(x,i)
    model = Evidence_Approximation(100.,100,)
    model.fit(X,y)
    evidence_list.append(model.evidence_function(X,y))
    #model = Bayesian_Regression(10,10)
    #model.fit(X,y)
    
    test = poly(xx,i)
    mean,sigma = model.predict(test)
    
    ax = fig.add_subplot(4,3,i+1)
    ax.scatter(x,y,c = "k",s=20)
    ax.plot(xx,func(xx),label = "True",c="b")
    ax.plot(xx,mean,label = f"pred, sigma={round(sigma.mean(),2)}",c = "r")
    ax.fill_between(xx, mean+sigma, mean-sigma,color='c', alpha=0.5, label="predict_std")

    ax.legend(loc=1)



