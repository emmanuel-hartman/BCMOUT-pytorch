import torch
import numpy as np
from bcmout.MetricSpace import MetricSpace 
from bcmout.Metric import Metric
from bcmout.LengthMetric import LengthMetric
from bcmout.Euclidean import EuclideanMetric

class Sphere(MetricSpace):
    """Class for a Sphere metric space
    
    Parameters
    ----------
    dim : int
        dimension of the Sphere
    metric : string  
        String specifying what type of metric to equip the space with
        Optional, default: 'Spherical'
    """
    
    def __init__(self, dim, metric='Spherical',**kwargs):
        if metric == 'Spherical':
            kwargs.setdefault("metric", SphericalMetric())
        else:
            kwargs.setdefault("metric", SphericalMetric())      
        self.shape= dim+1
        super().__init__(dim, **kwargs)
      
    def belongs(self, points, atol=1e-6):
        """Evaluate if a point belongs to the metric space.
        Parameters
        ----------
        point : array-like, shape=[point_shape,num_points]
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
        samples : array-like, shape=[point_shape,n_samples]
            Points sampled in the metric space.
        """
        
        points = 2*torch.rand(self.shape,samples)-1
        return points/torch.linalg.norm(points,dim=0)
        
    
class SphericalMetric(Metric):
    """Class for the Spherical metric object.
    """
        
    def __init__(self):
        super().__init__()    
    
    
    
class SphericalMetric(LengthMetric):
    """Class for the Spherical metric object
    """
    
    def __init__(self):
        super().__init__()
        
    def distance(self,point1,point2):
        """Compute the distance between two points.
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
        in_prod = torch.einsum('ia,ib->ab', point1,point2)
        d=torch.acos(in_prod)        
        d[torch.logical_and(torch.isnan(d),in_prod>0)] = 0    
        d[torch.logical_and(torch.isnan(d),in_prod<0)] = np.pi        
        return d
        
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
        w = torch.cross(torch.cross(point1,point2),point1)
        theta = np.arcos(torch.transpose(point1,0,1)*point2)
        for i in range(1,t+1):
            geodesic[i-1] = 
        return geodesic