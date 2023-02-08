import torch
import numpy as np
from MetricSpace import MetricSpace 
device = torch.device("cuda:0")


class Sphere(MetricSpace):
    """Class for a sphere
    """
    
    def __init__(self, dim):
        self.dim = dim
        self.shape= [dim + 1]
        super().__init__()
      
    def belongs(self, points, atol=1e-6):
        """Evaluate if a point belongs to the metric space.
        Parameters
        ----------
        point : array-like, shape=[num_points, point_shape]
            Point to evaluate.
        atol : float
            Absolute tolerance.
        Returns
        -------
        belongs : array-like, shape=[num_points]
            Boolean evaluating if point belongs to the metric space.
        """
        
        return torch.isclose(torch.linalg.norm(points,dim=0), torch.ones((points.shape[1])), atol=atol)
        
        
    def random(self, samples=1):
        """Sample random points on the metric space according to a uniform distribution.
        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        rand = array-like
            Random numbers to be used in the projection onto the hypersphere
        points = array-like
            Points on the sphere following projected from 
        Returns
        -------
        samples : array-like, shape=[n_samples, point_shape]
            Points sampled in the metric space.
        """
        
        points = torch.rand([samples] + self.shape)
        return points/torch.linalg.norm(points,dim=0)
    
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
        belongs : array-like, shape=[num_points1, num_points2, 1]
            Float evaluating the distance between two points in the metric space.
        """
        in_prod = torch.einsum('ia,ib->ab', point1,point2)
        d=torch.acos(in_prod)        
        d[in_prod>=1-1e-6] = 0        
        return d
