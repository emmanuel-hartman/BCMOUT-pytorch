import abc
import torch
import numpy as np
from bcmout.MetricSpace import MetricSpace 
from bcmout.Metric import Metric

class ConeOverM(MetricSpace):
    """ Class for the metric space defined by the cone over a metric space
    
    Parameters
    ----------
    M : Metric Space Object
        metric space that we are defining the cone over.
    delta : float64 
        parameter that defines the metric on the cone.
    metric : string  
        String specifying what type of metric to equip the space with
        Optional, default: None   
    """
    def __init__(self, M, delta, metric=None, **kwargs):
        
        if metric == "CosBar":
            kwargs.setdefault("metric", CosBarMetric(M,delta))
        elif metric == "Exp":
            kwargs.setdefault("metric", ExpMetric(M,delta))
        else:
            kwargs.setdefault("metric", CosBarMetric(M,delta))
            
        self.M=M
        self.shape = M.shape + 1 
        super().__init__( M.dim+1, **kwargs)
        
    def belongs(self,points,atol=1e-6):  
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
        Mpoints = points[1:,:]
        Rpoints = points[0:1,:]        
        Rbelongs = Rpoints >= 0
        Mbelongs = self.M.belongs(Mpoints)        
        return torch.logical_and(Rbelongs,Mbelongs)

    def random(self, samples=1, bound=1.0):
        """Sample random points on the metric space according to a uniform distribution.
        Parameters
        ----------
        samples : int
            Number of samples.
            Optional, default: 1
        bound : float64
            Bound for the cone weights we sample from 
            Optional, default: 1.0
        Returns
        -------
        points : array-like, shape=[point_shape,num_points]
            Points sampled in the metric space.
        """
        Mpoints = self.M.random(samples=samples)
        Rpoints = bound*torch.rand(1,samples)
        points = torch.concat([Rpoints,Mpoints], dim=0)
        return points
    
    def _energy(self,Mpoint1,Mpoint2):
        return self._metric._energy(Mpoint1,Mpoint2)
    
class CosBarMetric(Metric):    
    """Class for a Cone Space Metric object.
    
    Parameters
    ----------
    M : Metric Space Object
        metric space that we are defining the cone over.
    delta : float64 
        parameter that defines the metric on the cone.
    """
    def __init__(self,M,delta):
        self.M=M
        self.delta =delta
        super().__init__()
    
    def _energy(self,Mpoint1,Mpoint2):
        Mdist = self.M.distance(Mpoint1,Mpoint2)/(2*self.delta)
        Mdist[Mdist>np.pi/2] = 0
        return torch.cos(Mdist)

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
        distance : array-like, shape=[num_points1, num_points2]
            Float evaluating the distance between two points in the metric space.
        """        
        Mpoint1 = point1[1:,:]
        Rpoint1 = point1[0,:]  
        Mpoint2 = point2[1:,:]
        Rpoint2 = point2[0,:]  
                
        d= Rpoint1.reshape(-1,1)+Rpoint2
        d=d-2*torch.sqrt(torch.outer(Rpoint1, Rpoint2))*self._energy(Mpoint1,Mpoint2)     
        return torch.sqrt(4*(self.delta**2)*d)
    
class ExpMetric(Metric):
    """Class for a Cone Space metric.
    
    Parameters
    ----------
    M : Metric Space Object
        metric space that we are defining the cone over. 
    delta : float64 
        parameter that defines the metric on the cone.
    """     
    def __init__(self,M,delta):
        self.M=M
        self.delta =delta
        super().__init__()
    
    def _energy(self,Mpoint1,Mpoint2):
        Mdist = torch.exp(-1*self.M.distance(Mpoint1,Mpoint2)**2/(2*self.delta))
        return Mdist

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
        distance : array-like, shape=[num_points1, num_points2]
            Float evaluating the distance between two points in the metric space.
        """        
        Mpoint1 = point1[1:,:]
        Rpoint1 = point1[0,:]  
        Mpoint2 = point2[1:,:]
        Rpoint2 = point2[0,:]  
                
        d= Rpoint1.reshape(-1,1)+Rpoint2
        d=d-2*torch.sqrt(torch.outer(Rpoint1, Rpoint2))*self._energy(Mpoint1,Mpoint2)     
        return torch.sqrt(4*(self.delta**2)*d)
