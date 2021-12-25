
import numpy as np
from math import sqrt

class Ornstein_Uhlenbeck():   # action exploration noise..

    def __init__(self, sigma=1, mu=0, x=1, dt=1e-2, theta=0.15, step=1, stepsAvgSigma=100):
        # parameter decalration
        self.theta = theta
        self.stepsAvgSigma = stepsAvgSigma
        self.sigma = sigma
        self.mu = mu
        self.dt = dt              
        self.sigma_step = self.sigma / float(self.stepsAvgSigma)
        self.x = x
    
    def OU(self,step):
        sigma=  self.sigma_step / step
        x_new = self.x + self.theta*(self.mu - self.x)*self.dt + (sigma/sqrt(self.dt))*np.random.normal(size=1)
        self.x = x_new

        return x_new

   # a paramter giving from here..

a = Ornstein_Uhlenbeck()
print(a.OU(2))  

''' the output is the random values which are approximtely near to 1.
    which seems like a Gaussain Distribution. '''

     
