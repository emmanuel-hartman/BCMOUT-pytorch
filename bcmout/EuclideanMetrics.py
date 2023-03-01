import torch
import numpy as np
from bcmout.Metric import Metric
from bcmout.LengthMetric import LengthMetric

class EuclideanLengthMetric(LengthMetric):
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
        geodesic = torch.Tensor((point1.shape[0],t))
        for i in range(1,t+1):
            geodesic[i-1] = (1-i/(t+1))*point1 + (i/(t+1))*point2
        return geodesic

class EuclideanMetric(Metric):
    """Class for a Euclidean metric object.
    """
    
    def __init__(self):
        super().__init__()
        
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