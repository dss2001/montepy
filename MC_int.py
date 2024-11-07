"""
Filename: MC_int.py
Author: Daniel Santos Stone
Date: 31-10-2024
Description: Monte Carlo integrator for 1D functions
Contact: daniel.santos-stone@etu.univ-amu.fr 
Dependencies: seaborn, matplotlib, numpy

"""

#import libs
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from inspect import signature
from warnings import warn
from math import prod
#from tqdm import tqdm

#Monte Carlo with naive uniform sampler 

class MC_int():
    walks_arrays = []
    walks_funcs = []
    walks_averages = []
    average = None
    variance = None
    dim = None

    def __init__(self,N,M,func,lower,upper,scaling=1):
        """
        Class for numerically integrating 1D functions via Monte Carlo, with a naive uniform sampler.
        
        N: number of points to be generated for each replicate
        M: number of replicates to generate for Central Limit Theorem estimation
        func: function to be integrated
        lower: lower bound of integration
        upper: upper bound of integration
        scaling: determines how fine the grid of possibly generated points can be; the bigger, the finer. 
        
        """
        
        self.N=N
        self.M=M
        self.func=func
        self.lower=np.array(lower)
        self.upper=np.array(upper)
        self.scaling=scaling

    def get_func_dim(self):
        #get dimension of function
        sig = signature(self.func)
        self.dim = len(sig.parameters)
        try:
            assert len(self.upper) == len(self.lower)
        except:
            warn("Num of bounds do not match; upper has {up} bounds, and lower {low}".format(up=len(self.upper),low=len(self.lower)))
        try:
            assert self.dim > 0 
        except:
            warn("Num dimensions needs to be greater than zero")
        try:
            assert (self.dim == len(self.upper))    
            assert (self.dim == len(self.lower))
        except: 
            warn("Num dimensiones not equal to num bounds")

    @staticmethod
    def random_walk(N,func,lower,upper,scaling=1,dim=1):
        """
        Performs a random uncorrelated walk, by generating N random points, sampled from a uniform distribution, in the range [lower,upper]. 
        
        """

        vals = np.random.randint(lower*scaling,upper*scaling,size=(N,dim))*1/scaling
        funcs = np.array([func(*x) for x in vals])
        
        return prod([abs(up-low) for low,up in zip(lower,upper)])*np.average(funcs),vals,funcs  
    
    
    def calc(self):
        """
        Performs CLT estimation based on replicating the random walks, and then calculating the average, 
        and calculates the average and variances of the averages.
        
        """
        #get dimension of the function
        self.get_func_dim()

        for _ in range(0,self.M+1):
            average,vals,funcs = self.random_walk(N=self.N,func=self.func,lower=self.lower,upper=self.upper,scaling=self.scaling,dim=self.dim)

            self.walks_arrays.append(vals)
            self.walks_funcs.append(funcs)
            self.walks_averages.append(average)

        self.average = 1/len(self.walks_averages)*sum(self.walks_averages)
        self.variance = 1/len(self.walks_averages)*sum([(val-self.average)**2 for val in self.walks_averages])


    def plot_averages(self,bin_size=10,x_label="",save_name=None,**kwargs):

        fig,ax = plt.subplots(**kwargs)
        ax = sns.histplot(x=np.array(self.walks_averages),bins=bin_size,label=f"Var: {round(self.variance,3)} \nAvg: {round(self.average,3)}",alpha=0.5,linewidth=1.5)
        ax.legend(loc="upper right")
        ax.set_xlabel(x_label)
        plt.show()

        if save_name is not None:
            plt.savefig(save_name)
        
if __name__ == "__main__":
    #quick test
    para = lambda x,y: x**2*y**2

    mc = MC_int(10000,100000,para,[-4,-4],[4,4],scaling=100000)
    #avg,funcs,val = mc.random_walk(100,para,np.array([-4,-4,-4]),np.array([4,4,4]),100,3)
    #print(f"{avg}")
    mc.calc()
    avg = mc.average
    var = mc.variance
    print(f"Area: {avg} Var: {var}")
    #mc.plot_averages(save_name="test.png",x_label="Energies",bin_size=20)

    #area should be in the 40s; need to check the sampled points as well as if there is somethign else wrong

