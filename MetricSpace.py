import abc
import torch
from Metric import Metric

class MetricSpace(abc.ABC):
    """Class for metric space.
    Parameters
    ----------
    shape : tuple of int
        Shape of one element of the metric space.
        Optional, default : None.
    Attributes
    ----------
    metric : 
        metric on the metric space.
    shape : int
        Dimension of the array that represents the point.
    """

    def __init__(self, dim, metric=None, **kwargs):
        self.dim=dim
        self._metric=metric
        super().__init__(**kwargs)
        

    @abc.abstractmethod
    def belongs(self, points, atol):
        """Evaluate if a point belongs to the metric space.
        Parameters
        ----------
        point : array-like, shape=[..., *point_shape]
            Point to evaluate.
        atol : float
            Absolute tolerance.
        Returns
        -------
        belongs : array-like, shape=[...,]
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
        samples : array-like, shape=[..., *point_shape]
            Points sampled in the metric space.
        """
    
    def distance(self,points1,points2,**kwargs):
        return self._metric.distance(points1,points2, **kwargs)
        
    def dissimilarity(self,point1,**kwargs):
        return self.distance(point1,point1,**kwargs)
        
    @property
    def metric(self):
        """Metric associated to the space."""
        return self._metric

    @metric.setter
    def metric(self, metric):
        if metric is not None:
            if not isinstance(metric, Metric):
                raise ValueError("The argument must be a Metric object")
        self._metric = metric

