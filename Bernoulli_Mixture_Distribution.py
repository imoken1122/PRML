        
%%time
from scipy.misc import logsumexp
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets.mldata import fetch_mldata 

class Bernoulli_Mixture_Distribution:

    def __init__(self, cluster,max_iter):
        self.num_cluster = cluster
        self.max_iter = max_iter
    def Repare(self,X):
        self.N,self.D= X.shape
        self.pi =np.ones(self.num_cluster)/self.num_cluster
        self.mean = np.random.uniform(0.25, 0.75, size=(self.num_cluster, self.D))
        self.gamma = None
        
    def fit(self, X):
        self.Repare(X)
        for i in range(self.max_iter):
            params = np.hstack((self.pi.ravel(), self.mean.ravel()))
            self.E_step(X)
            self.M_step(X)
            if np.allclose(params,np.hstack((self.pi.ravel(), self.mean.ravel()))):
                break

    def Log_bernoulli(self, X):
        np.clip(self.mean, 1e-10, 1 - 1e-10, out=self.mean)
        return X.dot(np.log(self.mean).T) + (1 - X).dot(np.log(1 - self.mean).T)

    def E_step(self, X):
        temp= np.log(self.pi) + self.Log_bernoulli(X)
        temp -= np.log(np.sum(np.exp(temp), axis=1))[:,None]
        self.gamma = np.exp(temp)

    def M_step(self, X):
        Nk = np.sum(self.gamma, axis=0)
        self.pi = Nk / len(X)
        self.mean = (X.T.dot(self.gamma) / Nk).T

mnist = fetch_mldata('Mnist-original') 
x,y = mnist.data,mnist.target
train = []
for i in [9, 1, 2, 3, 8]:
    train.append(x[np.random.choice(np.where(y == i)[0], 200)])
    
train = np.concatenate(train, axis=0)
train = (train > 127).astype(np.float)

model = Bernoulli_Mixture_Distribution(5,10)
model.fit(train)

plt.figure(figsize=(20, 5))
for i, mean in enumerate(model.mean):
    plt.subplot(1, 5, i + 1)
    plt.imshow(mean.reshape(28, 28), cmap="gray")
    plt.axis('off')
plt.show()