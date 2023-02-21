import torch
from bcmout.MetricSpace import MetricSpace 
from bcmout.Metric import Metric


class Product(MetricSpace):
    """Class for a Euclidean Space
    """
    
    def __init__(self,M,N,a=1,**kwargs):
        kwargs.setdefault("metric", PythagoreanMetric(M,N,a))
        self.M=M
        self.N=N
        self.split=M.shape
        self.shape= M.shape+N.shape
        super().__init__(M.dim+N.dim, **kwargs)
      
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
        Mpoints = points[:self.split,:]
        Npoints = points[self.split:,:]  
        
        return torch.logical_and(self.M.belongs(Mpoints),self.N.belongs(Npoints))
        
        
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
        return torch.cat([self.M.random(samples),self.N.random(samples)],dim=0)

    
class PythagoreanMetric(Metric):    
    def __init__(self,M,N,a):
        self.M=M
        self.N=N
        self.a=a
        self.split=M.shape
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
        Mpoint1 = point1[:self.split,:]
        Mpoint2 = point2[:self.split,:]
        Npoint1 = point1[self.split:,:]
        Npoint2 = point2[self.split:,:]  
        
        dist1= self.M.distance(Mpoint1,Mpoint2)
        dist2= self.a*self.N.distance(Npoint1,Npoint2)
                
        return torch.sqrt(dist1**2+dist2**2)
    

