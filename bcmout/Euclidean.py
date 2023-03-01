import torch
import numpy as np
from bcmout.MetricSpace import MetricSpace 
from bcmout.EuclideanMetrics import *

class Euclidean(MetricSpace):
    """Class for a Euclidean metric space
    
    Parameters
    ----------
    dim : int
        dimension of the Euclidean space
    metric : string  
        String specifying what type of metric to equip the space with
        Optional, default: 'Euclidean'
    """
    
    def __init__(self, dim, metric="Euclidean",lengthMetric="Euclidean", **kwargs):
        if metric == "Euclidean":
            kwargs.setdefault("metric", EuclideanMetric())
        else:
            kwargs.setdefault("metric", EuclideanMetric())        
        if lengthMetric == "Euclidean":
            kwargs.setdefault("lengthMetric", EuclideanLengthMetric())
        else:
            kwargs.setdefault("lengthMetric", EuclideanLengthMetric())        
        self.shape=dim
        
        super().__init__(dim, **kwargs)
      
    def belongs(self, points, atol=1e-6):
        """Evaluate if a point belongs to the metric space.
        Parameters
        ----------
        points : array-like, shape=[point_shape,num_points]
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
        samples : int
            Number of samples.
            Optional, default: 1
        bound : float64
            Bound for the region to sample the points from
            Optional, default: 1.0
        Returns
        -------
        points : array-like, shape=[point_shape,samples]
            Points sampled in the metric space.
        """
        
        points = 2*bound*torch.rand(self.shape,samples)-1
        return points