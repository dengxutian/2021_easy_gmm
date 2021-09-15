# @Time : 2021/9/15 15:42
# @Author : Deng Xutian
# @Email : dengxutian@126.com


import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def generate_data():
    data = []
    for i in range(100000):
        x = random.random()
        y = random.random()
        z = random.random()
        x = x/math.sqrt(x**2 + y**2 + z**2)
        y = y/math.sqrt(x**2 + y**2 + z**2)
        z = z/math.sqrt(x**2 + y**2 + z**2)
        data.append(np.array([x, y, z]))
    data = np.array(data)
    np.save('data.npy', data)

class gmm():
    
    def __init__(self):
        self.k = 8
        self.x = np.load('data.npy')
        self.len = self.x.shape[0]
        self.dim = self.x.shape[1]
        self.mu = np.random.random((self.k, self.dim))
        self.var = np.repeat(np.array([np.diag(np.ones((self.dim, )))]), self.k, axis=0)
        self.pi = np.ones((self.k, )) / self.dim
        self.w = np.ones((self.len, self.k)) / self.dim
        self.logLH_list = []
    
    def step_e(self):
        self.update_w()
    
    def step_m(self):
        self.update_pi()
        self.update_mu()
        self.update_var()
    
    def save(self):
        np.save('mu.npy', self.mu)
        np.save('var.npy', self.var)
        np.save('pi.npy', self.pi)
        np.save('w', self.w)
        pass
    
    def load(self):
        self.mu = np.save('mu.npy')
        self.var = np.save('var.npy')
        self.pi = np.save('pi.npy')
        self.w = np.save('w')
        pass
    
    def update_w(self):
        pdfs = np.zeros(((self.len, self.k)))
        for i in range(self.k):
            pdfs[:, i] = self.pi[i] * multivariate_normal.pdf(self.x, self.mu[i], self.var[i])
        self.w = pdfs / pdfs.sum(axis=1).reshape(-1, 1)

    def update_pi(self):
        self.pi = self.w.sum(axis=0) / self.w.sum()

    def update_mu(self):
        for i in range(self.k):
            self.mu[i] = np.average(self.x, axis=0, weights=self.w[:, i])
    
    def update_var(self):
        for i in range(self.k):
            self.var[i] = np.cov(self.x - self.mu[i], rowvar=0, aweights=self.w[:, i])
            
    def logLH(self):
        pdfs = np.zeros(((self.len, self.k)))
        for i in range(self.k):
            pdfs[:, i] = self.pi[i] * multivariate_normal.pdf(self.x, self.mu[i], self.var[i])
        logLH = np.mean(np.log(pdfs.sum(axis=1)))
        self.logLH_list.append(logLH)
        
    def gmr(self, x):
        belta = np.zeros((self.k,))
        xi = np.zeros((self.k, 2))
        for i in range(self.k):
            mu_t = self.mu[i][0:1]
            mu_s = self.mu[i][1:3]
            var_tt = self.var[i][0:1, 0:1]
            var_st = self.var[i][0:1, 1:3].transpose()
            var_ss = self.var[i][1:3, 1:3]
            belta[i] = multivariate_normal.pdf(x, mu_t, var_tt)
            var = np.dot(var_st, np.linalg.inv(var_tt))
            xi[i] = mu_s + np.dot(var, x - mu_t)
        belta = belta / belta.sum()
        y = np.average(xi, axis = 0, weights = belta)
        return y

if __name__ == '__main__':
    generate_data()
    demo = gmm()
    for i in range(100):
        print(i + 1)
        demo.step_e()
        demo.step_m()
        demo.logLH()

    vec = np.array(demo.logLH_list)
    plt.plot(np.arange(0, vec.shape[0]), vec)
    plt.show()

    # x = 0.8
    # arr = demo.gmr(np.array([x]))
    # y = arr[0]
    # z = arr[1]
    # print(x, y, z)
    # print(x**2 + y**2 + z**2)

    demo.save()