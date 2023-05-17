import torch.nn as nn
import torch

class SequentialFlow(nn.Module):
    """A generalized nn.Sequential container for normalizing flows.
    """

    def __init__(self, layersList):
        super(SequentialFlow, self).__init__()
        self.chain = nn.ModuleList(layersList)

    def forward(self, x, logpx=None, reverse=False, inds=None):
        if inds is None:
            if reverse:
                inds = range(len(self.chain) - 1, -1, -1)
            else:
                inds = range(len(self.chain))

        if logpx is None:
            for i in inds:
                x = self.chain[i](x, reverse=reverse)
            return x
        else:
            for i in inds:
                x, logpx = self.chain[i](x, logpx, reverse=reverse)
            return x, logpx
            
class SequentialFlowAug(nn.Module):
    """A generalized nn.Sequential container for normalizing flows.
    """

    def __init__(self, layersList):
        super(SequentialFlowAug, self).__init__()
        self.chain = nn.ModuleList(layersList)
        self.aug_hist = [i for i in range(len(self.chain))]

    def pit_stop(self, x, ind=0, reverse=False):
        if reverse:
            x[:,:x.shape[1]//2]=self.aug_hist[ind][0].repeat(x.shape[0],1,1,1)
            
            return x
        else:
            self.aug_hist[ind], x= x[:,:x.shape[1]//2], x[:,x.shape[1]//2:]
            x = torch.cat((torch.zeros(x.shape).to(x),x),1)
            
            return x
            
            
    def forward(self, x, logpx=None, reverse=False, inds=None):

        if inds is None:
            if reverse:
                inds = range(len(self.chain) - 1, -1, -1)
            else:
                inds = range(len(self.chain))

        if logpx is None:
            for i in inds:
                if reverse:
                    x = self.pit_stop(x, ind=i, reverse=reverse)
                    x = self.chain[i](x, reverse=reverse)
                else:                
                    x = self.chain[i](x, reverse=reverse)
                    x = self.pit_stop(x, ind=i, reverse=reverse)
            return x
                    
        else:
            for i in inds:
                
                if reverse:
                    x = self.pit_stop(x, ind=i, reverse=reverse)                
                    x, logpx = self.chain[i](x, logpx, reverse=reverse)
                else:              
                    x, logpx = self.chain[i](x, logpx, reverse=reverse)         
                    x = self.pit_stop(x, ind=i, reverse=reverse)             
                    
            return x, logpx
            

