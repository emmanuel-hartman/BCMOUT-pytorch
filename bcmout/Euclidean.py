import torch
import numpy as np
from bcmout.MetricSpace import MetricSpace 
from bcmout.Metric import Metric


class Euclidean(MetricSpace):
    """Class for a Euclidean Space
    """
    
    def __init__(self, dim, **kwargs):
        kwargs.setdefault("metric", EuclideanMetric())
        self.shape= dim
        super().__init__(dim, **kwargs)
      
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
        
        return (points.shape[0]==self.shape)*torch.ones(points.shape[1], dtype=torch.bool)
        
        
    def random(self,samples=1,bound=1.0):
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
        
        points = 2*bound*torch.rand(self.shape,samples)-1
        return points

    
class EuclideanMetric(Metric):    
    def __init__(self):
        super().__init__()
        
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
        
        diffs= point1.reshape(point1.shape[0],1,point1.shape[1])-point2.reshape(point2.shape[0],point2.shape[1],1)          
        return torch.sqrt((diffs**2).sum(dim=0))
    

