import torch
import numpy as np
import ConeSpace
import abc
device = torch.device("cuda:0")


class MeasuresSpace:
    """Class for space of measures over a metric space M. Should allow comparison"""

    def __init__(self, M):
        """
        Parameters
        ----------
        M : MetricSpace
            Space which the measures are over
        """
        self.M = M
        self.dim = M.dim+1
        self.shape = M.shape + 1
        
    def random(self, n_supports=1, maxWeight=1):
        """Create some random measures over a metric space M according to a uniform distribution.
        Parameters
        ----------
        """
        
        supports = self.M.random(n_supports)
        masses = maxWeight*torch.rand(1,n_supports)
        point = torch.concat([masses,supports], dim=0)
        
        return point
                

    def belongs(self, points):
        """Checks whether or not two or more measures are over the same base space M"""
        
        
        return 
    

    def distance(self, point1, point2):
        """Here we have our algorithm"""
        
        return

    
