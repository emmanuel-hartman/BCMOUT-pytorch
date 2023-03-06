import torch
import numpy as np
from bcmout.MetricSpace import MetricSpace 
from bcmout.Metric import Metric
from bcmout.LengthMetric import LengthMetric

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
    
    def __init__(self, dim, metric="Euclidean", **kwargs):
        if metric == "Euclidean":
            kwargs.setdefault("metric", EuclideanMetric())
        else:
            kwargs.setdefault("metric", EuclideanMetric())
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

class EuclideanMetric(LengthMetric):
    """Class for a Euclidean length metric object.
    """
    
    def __init__(self):
        super().__init__()
        
    def geodesic(self,point1,point2,t):
        """Returns a set of t points equally spaced on the geodesic between point1 and point2.
        Parameters
        ----------
        point1 : array-like, shape=[point_shape,num_points1]
            Point to evaluate.
        point2 : array-like, shape=[point_shape,num_points2]
            Point to evaluate.
        Returns
        -------
        geodesic : array-like, shape=[dim, t]
            t points on the geodesic between point1 and point2
        """
        diff=(point2-point1)/(t-1)
        return torch.cat([torch.unsqueeze((point1+diff*i),0) for i in range(0,t)],dim=0)
    
    def distance(self,points1,points2):
        """Compute the distance between two points.
        Parameters
        ----------
        points1 : array-like, shape=[point_shape,num_points1]
            Point to evaluate.
        points2 : array-like, shape=[point_shape,num_points2]
            Point to evaluate.
        Returns
        -------
        belongs : array-like, shape=[num_points1, num_points2, 1]
            Float evaluating the distance between two points in the metric space.
        """
        diffs= points1.reshape(points1.shape[0],1,points1.shape[1])-points2.reshape(points2.shape[0],points2.shape[1],1)       
        return torch.sqrt((diffs**2).sum(dim=0))
        
    
    
class pMetric(Metric):
    """Class for a metric induced by the p-norm (for p>1).
    """
    
    def __init__(self, p, **kwargs):
        self.p = p
        super().__init__()
        
    def distance(self,point1,point2):
        """Compute the distance between two points induced by the p-norm.
        Parameters
        ----------
        point1 : array-like, shape=[point_shape,num_points1]
            Point to evaluate.
        point2 : array-like, shape=[point_shape,num_points2]
            Point to evaluate.
        Returns
        -------
        belongs : array-like, shape=[num_points1, num_points2, 1]
            Float evaluating the distance between two points in the metric space.
        """
        diffs= point1.reshape(point1.shape[0],1,point1.shape[1])-point2.reshape(point2.shape[0],point2.shape[1],1)          
        return torch.pow((diffs**p).sum(dim=0),1/p)