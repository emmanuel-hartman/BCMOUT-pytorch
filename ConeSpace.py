import abc
import torch
import numpy as np
from MetricSpace import MetricSpace 
device = torch.device("cuda:0")

class ConeOverM(MetricSpace):
    def __init__(self, M, delta):
        """
        Parameters
        ----------
        M : Metric Space
        Metric space we want to define the cone over.
        ---------
        dim : integer
        Dimension of the cone (one dimension higher than the metric space M)
        ---------
        shape : list
        The shape of a tensor representing an element of the cone over M
        ---------
        delta : float64
        Parameter defining the metric on the cone (corresponding to the "width" of the cone)"""
        self.M=M
        self.dim = M.dim+1
        self.shape = [1] + M.shape
        self.delta = delta 
        
    def belongs(self,points,atol=1e-6):
        """
        Parameters
        ---------
        points : list
        A list of points of the cone
        ---------
        atol : float64
        Tolerance for comparison of points
        """
        Mpoints = points[1:,:]
        Rpoints = points[0:1,:]        
        Rbelongs = Rpoints >= 0
        Mbelongs = self.M.belongs(Mpoints)        
        return torch.logical_and(Rbelongs,Mbelongs)

    def random(self, samples=1, maxWeight=1):
        """
        Parameters
        ---------
        sampes : int
        Number of samples to be taken
        ---------
        maxWeight : float64
        Maximum weight for any support
        ---------
        Returns
        ---------
        points : tensor
        Tensor of random weights (up to maxWeight) on random supports from the metric space M
        """
        Mpoints = self.M.random(samples)
        Rpoints = maxWeight*torch.rand(1,samples)
        points = torch.concat([Rpoints,Mpoints], dim=0)
        return points
    
    def _energy(self,Mpoint1,Mpoint2):
        """
        Helper function in computing the distance on the cone space. 
        Takes the cos of the distance of the M components over 2*delta and sends negative values to 0.
        Parameters
        ---------
        Mpoint1 : tensor
        Point belonging to M
        ---------
        Mpoint2 : tensor
        Point belonging to M
        """
        
        Mdist = self.M.distance(Mpoint1,Mpoint2)/(2*self.delta)
        Mdist[Mdist>np.pi/2] = 0
        return torch.cos(Mdist)

    def distance(self,point1,point2):
        """Compute the distance between two points on the cone. Separates a point into its M component and its measure component and then computes the distance using delta and the inherited metric M.distance.
        Parameters
        ----------
        point1 : array-like
            Point to evaluate.
        point2 : array-like
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
        return torch.sqrt(2*(self.delta**2)*d)
