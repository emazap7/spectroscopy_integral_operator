import math
import numpy as np
from IPython.display import clear_output
from tqdm import tqdm_notebook as tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib as mpl
import matplotlib.cm as cm

import torch
from torch import Tensor
from torch import nn
from torch.nn  import functional as F
from torch.autograd import Variable
import random
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from torchcubicspline import(natural_cubic_spline_coeffs, NaturalCubicSpline)


if torch.cuda.is_available():  
    device = "cuda:0" 
else:  
    device = "cpu"



class integral_operator(nn.Module):
    def __init__(self, 
                 dim,
                 x,
                 G,
                 integrator,
                 num_internal_points=100,
                 mc_samplings=5000,
                 upper_bound=None,
                 lower_bound=None,
                 integration_dim=-2):
        super(integral_operator, self).__init__()

        self.dim = dim
        self.mc_samplings = mc_samplings
        self.x = x.to(device)
        self.x_d = torch.linspace(0,1,num_internal_points).to(device)
        if lower_bound is None:
                lower_bound = lambda x: self.x[0]
        if upper_bound is None:
            upper_bound = lambda x: self.x[-1]
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.G = G
        self.integrator = integrator
        self.integration_dim = integration_dim

    def _interpolate_y(self, y: torch.Tensor):
        
        x=self.x_d
        
        y = y.to(device)
        
        coeffs = natural_cubic_spline_coeffs(x, y)
        interpolation = NaturalCubicSpline(coeffs)
        
        def output(point:torch.Tensor):
            return interpolation.evaluate(point.to(device))
        
        return output


    def forward(self,y,x):

        interpolated_y = self._interpolate_y(y.requires_grad_(True))
        def integral(x):
            
            x = x.reshape(y.shape[0],1,1).repeat(1,self.mc_samplings,1)
            
            def integrand(s):
                s = s.squeeze(-1).requires_grad_(True).to(device)
                
                out = self.G.forward(
                                    interpolated_y(s[:]),
                                    x,
                                    s.reshape(1,self.mc_samplings,1).repeat(y.shape[0],1,1)
                                    )
                
                return out
                
            ####
            if self.lower_bound(x) < self.upper_bound(x):
                interval = [[self.lower_bound(x),self.upper_bound(x)]]
            else: 
                interval = [[self.upper_bound(x),self.lower_bound(x)]]
            ####
            
            
            return self.integrator.integrate(
                           fn= lambda s: torch.sign(self.upper_bound(x)-self.lower_bound(x))*integrand(s)[:,:,:self.dim],
                           dim= 1,
                           N=self.mc_samplings,
                           integration_domain = interval,
                           out_dim = self.integration_dim,
                           )
        return integral(x)

        