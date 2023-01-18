import torch
import numpy as np
from ConeSpace import ConeOverM
import abc
device = torch.device("cuda:0")


class MeasuresSpace:
    """Class for space of measures over a metric space M. Should allow comparison"""

    def __init__(self, M, delta):
        """
        Parameters
        ----------
        M : MetricSpace
            Space which the measures are over
        """
        self.M = M
        self.CoM = ConeOverM(M,delta)
        self.dim = M.dim+1
        self.shape = -1
        
    def random(self, n_samples=1, n_supports=1, maxWeight=1):
        """Create some random measures over a metric space M according to a uniform distribution.
        Parameters
        ----------
        """
        ls=[]
        for i in range(0,n_samples):
            supports = self.M.random(n_supports)
            masses = maxWeight*torch.rand(1,n_supports)
            point = torch.concat([masses,supports], dim=0)
            ls+=[point]
        
        return ls
                

    def belongs(self, points):
        """Checks whether or not two or more measures are over the same base space M"""
        belongs = torch.zeros((len(points)), dtype=torch.bool)
        for i in range(0,len(points)):
            point=points[i]
            belongs[i]=torch.all(self.CoM.belongs(point))
        return belongs
    
    def _pairwise_distance(self,point1,point2):
        point1=point1.to(device)
        point2=point2.to(device)
        a = point1[0,:]
        b = point2[0,:]
        u = point1[1:,:]
        v = point2[1:,:]
        Omega1 = self.CoM._energy(u,v)
        Omega = torch.zeros((Omega1.shape[0]+1,Omega1.shape[1]+1)).to(device)
        Omega[1:,1:]= Omega1

    def distance(self, point1, point2):
        """Here we have our algorithm"""
        distance = torch.zeros((len(point1),len(point2)))
        for i in range(0, distance.shape[0]):
            for j in range(0,distance.shape[1]):
                distance[i,j]=self._pairwise_distance(point1[i], point2[j])
        return distance

    
