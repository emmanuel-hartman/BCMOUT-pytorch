import abc
import torch
import numpy as np
from MetricSpace import MetricSpace 
device = torch.device("cuda:0")

class ConeOverM(MetricSpace):
    """
    Parameters
    ----------
    M : Metric Space
        metric space that we are defining the cone over.
    delta : float64 
        parameter that defines the metric on the cone
        
    """
    def __init__(self, M, delta):
        self.M=M
        self.dim = M.dim+1
        self.shape = M.shape + 1
        self.delta = delta 
        
    def belongs(self,points,atol=1e-6):        
        Mpoints = points[1:,:]
        Rpoints = points[0:1,:]        
        Rbelongs = Rpoints >= 0
        Mbelongs = self.M.belongs(Mpoints)        
        return torch.logical_and(Rbelongs,Mbelongs)

    def random(self, samples=1, maxWeight=1):
        Mpoints = self.M.random(samples)
        Rpoints = maxWeight*torch.rand(1,samples)
        points = torch.concat([Rpoints,Mpoints], dim=0)
        return points
    
    def _energy(self,Mpoint1,Mpoint2):
        Mdist = self.M.distance(Mpoint1,Mpoint2)/(2*self.delta)
        Mdist[Mdist>np.pi/2] = 0
        return torch.cos(Mdist)

    def distance(self,point1,point2):
        """Compute the distance between two points.
        Parameters
        ----------
        point1 : array-like, shape=[num_points1, point_shape]
            Point to evaluate.
        point2 : array-like, shape=[num_points2, point_shape]
            Point to evaluate.
        Returns
        -------
        belongs : array-like, shape=[num_points1, num_points2]
            Float evaluating the distance between two points in the metric space.
        """        
        Mpoint1 = point1[1:,:]
        Rpoint1 = point1[0,:]  
        Mpoint2 = point2[1:,:]
        Rpoint2 = point2[0,:]  
                
        d= Rpoint1.reshape(-1,1)+Rpoint2
        d=d-2*torch.sqrt(torch.outer(Rpoint1, Rpoint2))*self._energy(Mpoint1,Mpoint2)     
        return 2*(self.delta**2)*d
