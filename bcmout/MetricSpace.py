import abc
import torch
from bcmout.Metric import Metric

class MetricSpace(abc.ABC):
    """Class for metric space.
    Parameters
    ----------
    dim : int
        dimension of the metric space
    metric:
        Metric object associated with the metric space
        
    Attributes
    ----------
    _metric : 
        metric on the metric space.
    dim : int
        Dimension of the array that represents the point.
    """

    def __init__(self, dim, metric=None, lengthMetric=None, **kwargs):
        self.dim=dim
        self._metric=metric
        self._lengthMetric=lengthMetric
        super().__init__(**kwargs)
        

    @abc.abstractmethod
    def belongs(self, points, atol):
        """Evaluate if a point belongs to the metric space.
        Parameters
        ----------
        point : array-like, shape=[point_shape,num_points]
            Point to evaluate.
        atol : float
            Absolute tolerance.
        Returns
        -------
        belongs : array-like, shape=[num_points,]
            Boolean evaluating if point belongs to the metric space.
        """
        
    @abc.abstractmethod
    def random(self, samples=1):
        """Sample random points on the metric space according to some distribution.
        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        Returns
        -------
        samples : array-like, shape=[point_shape,num_points]
            Points sampled in the metric space.
        """
    
    def distance(self,points1,points2,**kwargs):
        return self._metric.distance(points1,points2, **kwargs)
    
    def geodesic(self,point1,point2,t,**kwargs):
        return self._lengthMetric.distance(point1,point2,t,**kwargs)
        
    def dissimilarity(self,points1,**kwargs):
        return self.distance(points1,points1,**kwargs)