from torch import nn
import torch
import numpy as np

if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"
device = torch.device(dev)




class F_NN(nn.Module):
    def __init__(self,in_dim,out_dim,shapes,NL=nn.ELU):
        super(F_NN, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_layers = len(shapes) - 1
        self.shapes = shapes
        self.first = nn.Linear(in_dim,shapes[0])
        self.layers = nn.ModuleList([nn.Linear(shapes[i],shapes[i+1]) for i in range(self.n_layers)])
        self.last = nn.Linear(shapes[-1], out_dim)
        self.NL = NL(inplace=True) 
        
    def forward(self, y):
        y_in = y
        y = y_in.flatten(-2,-1)
        y = self.NL(self.first.forward(y))
        for layer in self.layers:
            y = self.NL(layer.forward(y))   
        y = self.last.forward(y)
        
        return y.view(y_in.shape[0],y_in.shape[1],y_in.shape[2]-1,y_in.shape[3])


class NN_feedforward(nn.Module):
    def __init__(self,in_dim,out_dim,shapes,NL=nn.ELU):
        super(NN_feedforward, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_layers = len(shapes) - 1
        self.shapes = shapes
        self.first = nn.Linear(in_dim,shapes[0])
        self.layers = nn.ModuleList([nn.Linear(shapes[i],shapes[i+1]) for i in range(self.n_layers)])
        self.last = nn.Linear(shapes[-1], out_dim)
        self.NL = NL(inplace=True) 
        
    def forward(self, y):
        y_in = y
        y = self.NL(self.first.forward(y_in))
        for layer in self.layers:
            y = self.NL(layer.forward(y))   
        y = self.last.forward(y)
        
        return y


class G_global(nn.Module):
    def __init__(self,in_dim,out_dim,shapes,NL=nn.ELU):
        super(G_global, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_layers = len(shapes) - 1
        self.shapes = shapes
        self.first = nn.Linear(in_dim+2,shapes[0])
        self.layers = nn.ModuleList([nn.Linear(shapes[i],shapes[i+1]) for i in range(self.n_layers)])
        self.last = nn.Linear(shapes[-1], out_dim)
        self.NL = NL(inplace=True) 
        
    def forward(self, y, t, s):
        y = y
        y_in = torch.cat([y,t,s],-1)
        y = self.NL(self.first.forward(y_in))
        for layer in self.layers:
            y = self.NL(layer.forward(y))   
        y = self.last.forward(y)

        return y


class ConvNeuralNet1D(nn.Module):
    def __init__(self,dim,hidden_dim=32,hidden_ff= 64,out_dim=32,K1=[4,4],K2=[4,4]):
        super(ConvNeuralNet1D, self).__init__()
        self.conv_layer1 = nn.Conv1d(dim, hidden_dim,
                                     kernel_size=[K1[0]],
                                     stride=K1[1]
                                    )
        
        self.conv_layer2 = nn.Conv1d(hidden_dim, out_dim,
                                     kernel_size=[K2[0]],
                                     stride=K2[1]
                                    )
        self.fc1 = nn.Linear(out_dim, hidden_ff)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_ff, out_dim)
        
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = out.permute(0,2,1)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = out.permute(0,2,1)
        return out
