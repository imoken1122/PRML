import matplotlib.pyplot as plt
import numpy as np
class Variationa_Linear_Regression:
    def __init__(self,max_iter,degree):
        self.beta=beta
        self.iter = max_iter
        self.degree = degree
        
    #def Poly(self,X):
    #    return np.array([[x ** i for i in range(self.degree+1)] for x in X])
    
    def Gaussian_basis(self,X):
        bases = np.linspace(-1, 1, 5)
        return np.array([np.exp(- (X - b) ** 2. * 10.) for b in bases]).T

    def Repear(self,X):
        phi = self.Gaussian_basis(X)
        self.N,self.M = phi.shape
        self.m_N = np.zeros([self.M,1])
        self.S_N = np.eye(self.M)
        
        self.alpha_N,self.beta_N =.1,.1
        self.c_M,self.d_M =.1,.1
        self.E_alpha,self.E_beta = 0.001,100
        self.a0,self.b0 = 10.,10.
        self.c0,self.d0 = 1.,1.
        return phi
    
    def fit(self,X,t):
        Phi = self.Repear(X)
        
        for i in range(self.iter):
            params = np.hstack([self.E_alpha, self.E_beta])
            print(f"{i} : beta{self.E_beta}, alpha{self.E_alpha}")
            self.q_w(Phi,t)
            self.q_alpha()
            self.q_beta(Phi,t)
            
            if np.allclose(params,np.hstack([self.E_alpha, self.E_beta])):
                break
                
    def q_w(self,phi,t):
        S_inv = self.E_alpha*np.eye(self.M) + self.E_beta*(phi.T.dot(phi))
        self.S_N = np.linalg.inv(S_inv)
        self.m_N = (self.E_beta * self.S_N.dot(phi.T.dot(t))).reshape(-1,1)
    
        
    def q_alpha(self):
        E_ww = (self.m_N.T.dot(self.m_N) + np.trace(self.S_N)).flatten()
        self.alpha_N = self.a0 + self.M/2.
        self.beta_N = self.b0 + 0.5 * E_ww
        self.E_alpha = self.alpha_N/self.beta_N
    
    def q_beta(self,phi,t):

        self.c_M = self.c0 + 0.5*self.N
        a = (t.reshape(-1,1) - phi.dot(self.m_N))
        self.d_M = (self.d0 + 0.5*a.T.dot(a) \
                + np.trace(phi.T.dot(phi).dot(self.S_N))).flatten()
        self.E_beta = self.c_M/self.d_M
    
    def Predictive_distribution(self,X,t):
        phi = self.Gaussian_basis(X)
        
        mu = phi.dot(self.m_N)
        sigma = np.sqrt(1/self.E_beta + phi.dot(self.S_N.dot(phi.T)))
        
        return mu.flatten(),np.diag(sigma)


def func(x):
    return np.sin(np.pi*x)

def dataset(size):
    np.random.seed(70)
    x = np.random.uniform(-1.5, 1.5, size)
    t = func(x)
    return x,t+np.random.normal(0,0.6,size)

X,t = dataset(20)
model = Variationa_Linear_Regression(10,3)
model.fit(X,t)


xx = np.linspace(-2,2,300)
tt = func(xx)
pred_mu,pred_sigma=model.Predictive_distribution(xx,t)
plt.plot(xx,tt,label = "True_func",alpha=0.5,linestyle="--",c="r")
plt.scatter(X,t,label="data_point",c="k")
plt.xlim([-2,2])
plt.ylim([-2,2])
plt.fill_between(xx, pred_mu + pred_sigma, pred_mu- pred_sigma,color="C9",label="predict_sigma", alpha=.1)
plt.plot(xx,pred_mu,c="C9",label = "predict_mean")

plt.legend(loc='lower right')
       