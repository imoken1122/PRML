import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


class EM_Algorithm():
    def __init__(self,max_iter,num_cluster,num_data):
        self.max_iter = max_iter
        self.num_cluster = num_cluster
        self.num_data = num_data
        
        
    def Repare(self,X):
        self.N,self.M= X.shape
        self.pi = np.ones(self.num_cluster)/self.num_cluster
        self.mean = np.random.rand(self.num_cluster,self.M)
        self.cov = np.zeros( (self.num_cluster,self.M,self.M))
        for k in range(self.num_cluster):
            self.cov[k] = [[1.,0.],[0.,1.]]
        self.gamma = np.zeros((self.N,self.num_cluster))
    
    def StandardScaler(self,X):
        mean = np.mean(X,axis=0)
        std = np.std(X,axis = 0)
        for i in range(self.M):
            X[:,i] = (X[:,i] - mean[i])/std[i]
        return X
    
    def fit(self,X):
        self.Repare(X)
        self.iter = 0
        X = self.StandardScaler(X)

        for i in range(self.max_iter):
            old_params = np.hstack((self.pi.ravel(), self.mean.ravel(), self.cov.ravel()))
            old = self.mean
            self.E_step(X)
            self.M_step(X)
            self.Visualizer(X,old,self.mean,i)
            if np.allclose(old_params,np.hstack((self.pi.ravel(), self.mean.ravel(), self.cov.ravel()))):
                break
            
            
    def E_step(self,X):

        for n in range(self.N):
            denom = 0.0
            for j in range(self.num_cluster):
                denom += self.pi[j] * self.Gaussian(X[n],self.mean[j],self.cov[j])
            for k in range(self.num_cluster):
                self.gamma[n][k] = self.pi[k] * self.Gaussian(X[n],self.mean[k],self.cov[k])/denom
   

    def M_step(self,X):    
        
        for k in range(self.num_cluster):
            N_k = 0.0
            for n in range(self.N):
                N_k += self.gamma[n][k]
                
            self.mean[k] = np.array([0.,0.])
            for n in range(self.N):
                self.mean[k] += self.gamma[n][k] * X[n]
            self.mean[k] /= N_k
            
            self.cov[k] = np.array([[0.,0.],[0.,0.]])
            for n in range(self.N):
                sub = (X[n] - self.mean[k]).reshape(2,1)
                self.cov[k] += self.gamma[n][k] *np.dot(sub,sub.T)
            self.cov[k] /= N_k
            
            self.pi[k] = N_k/self.N

        
    def Likelihood(self,X,mean,cov,pi):
        likelihood = 0.0
        for n in range(self.N):
            sub = 0.0
            for k in range(self.num_cluster):
                sub += pi[k] * self.Gaussian(X[n],mean[k],cov[k])
            likelihood += np.log(sub)
        return likelihood
    
    def Gaussian(self,x, mean, cov):
        normal = (1 / ((2 * np.pi) ** (x.size/2.0)))*(1 / (np.linalg.det(cov) ** 0.5))
        exp = - 0.5 * np.dot(np.dot(x - mean, np.linalg.inv(cov)), x - mean)
        return normal* np.exp(exp)
    
    def Visualizer(self,data,mu_prev,mu,f):
        c = ['r', 'g', 'b']
        for i in range(self.N):
            plt.scatter(data[i,0], data[i,1], s=30, c=self.gamma[i], alpha=0.5, marker="+")
        for i in range(self.num_cluster):
            plt.scatter([mu_prev[i, 0]], [mu_prev[i, 1]], c=c[i], marker='o', alpha=0.8)
            plt.scatter([mu[i, 0]], [mu[i, 1]], c=c[i], marker='o', edgecolors='k', linewidths=1)
        plt.title("step:{}".format(f))


def Make_Dataset():
    
    n = [100, 100, 100]
    N = np.sum(n)
    mu_true = np.array(
         [[1,10],
          [2, 9.8],
          [2.8,10.1]])
    K,D = mu_true.shape
    sigma_true = np.array(
            [[[0.1,  0.085],[ 0.085, .1]],
              [[0.1, -0.085],[-0.085, 0.1]],
              [[0.1,  0.085],[ 0.085, 0.1]]
            ])
    all_data = None
    for i in range(3):
        if all_data is None:
            all_data = st.multivariate_normal.rvs(mean=mu_true[0], cov=sigma_true[0], size=n[i])
        else:
            all_data = np.r_[all_data,st.multivariate_normal.rvs(mean=mu_true[i], cov=sigma_true[i], size=n[i])]
    return all_data,n

data,data_n = Make_Dataset()
model = EM_Algorithm(10,3,data_n)
model.fit(data)
plt.show()

