import abc
import torch
device = torch.device("cuda:0")

class ConeOverM(MetricSpace):
    """
    Parameters
    ----------
    dim : int
        Dimension of the cone space (dimension of the manifold plus one). 
    shape : tuple of int
        Shape of one element of M 
    """
    def __init__(self, M, delta):
        self.M = M
        self.dim = M.dim + 1
        self.delta = delta
        self.shape = ???? 
        return
    
    def belongs(self,point,atol):        
        return 

    def randomPoint(self, samples=1):
        for i in samples:

            return


    def distance(self, point1, point2):
        

        return
