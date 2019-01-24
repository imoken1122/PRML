
import numpy as np
from matplotlib import cm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

class Multiclass_logisticregression:
    def __init__(self,max_iter,mu,batch,random_seed
                ):
        self.max_iter = max_iter
        self.seed = np.random.RandomState(random_seed)
        self.w_ = None
        self.mu = mu
        self.batch_size = batch
        self.loss=[]
        self.score=[]
        
    def fit(self,x,T):
        K = T.shape[1]
        N,M = x.shape
        x = np.c_[x,np.ones([N,1])] # N * M+1
        self.w_ = self.seed.rand(M+1,K) # M+1 * K 
        batch_num = int(N/self.batch_size)
        
        for i in range(self.max_iter):
            for j in range(batch_num):
                x_batch = x[j*self.batch_size:self.batch_size*(j+1),:]
                t_batch = T[j*self.batch_size:self.batch_size*(j+1),:]
                self.grad_E = self.cross_entropy_grad(x_batch,t_batch)
                self.w_ -= self.mu*self.grad_E
            self.loss.append(self.closs_entropy(x, T))
            self.score.append(self.acc_score(T, self.predict(x)))
            
    def softmax(self,a):
        return np.exp(a)/np.sum(np.exp(a), axis=1)[:, np.newaxis]
    
    def cross_entropy_grad(self,X,T):
        a = X.dot(self.w_) # N * K 
        Y = self.softmax(a)
        dE = np.dot(X.T, Y -T)/X.shape[0]
        return dE
    
    def closs_entropy(self, X, T):
        Y= self.softmax(np.dot(X, self.w_) )
        return -np.sum(T*np.log(Y))/X.shape[0]
    
    def predict(self, X):
        Y = self.softmax(np.dot(X, self.w_))
        label = np.argmax(Y, axis=1)        
        pred = np.zeros([X.shape[0], self.w_.shape[1]])
        for i in range(len(pred)):
            pred[i, label[i]] = 1
        return pred
    
    def acc_score(self, true, pred):
        acc = np.array([np.sum(true[i,:]==pred[i,:])==len(true[i,:]) for i in range(len(true))])
        return np.sum(acc)/len(acc)
    


K=2
batch = 20
num_epoch = 500
mu = 0.001
data = load_iris()
X = data.data[:100,1:3]
t = data.target[:100]
Y = np.eye(K)[t]
trainx,testx,trainy,testy = train_test_split(X,Y,random_state = 0,shuffle=True)

model = Multiclass_logisticregression(num_epoch,mu,batch,0)
model.fit(trainx,trainy)

testx = np.c_[testx,np.ones([testx.shape[0],1])] 
pred = model.predict(testx)
model.acc_score(testy,pred)
