import abc
import torch
device = torch.device("cuda:0")

class MetricSpace(abc.ABC):
    """Class for metric space.
    Parameters
    ----------
    shape : tuple of int
        Shape of one element of the metric space.
        Optional, default : None.
    Attributes
    ----------
    distance : 
        distance on the metric space.
    shape : int
        Dimension of the array that represents the point.
    """

    def __init__(self, **kwargs):
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

    @abc.abstractmethod
    def distance(self, points):
        """Compute the distance between two or more points.
        Parameters
        ----------
        point1 : array-like, shape=[..., *point_shape]
            Point to evaluate.
        point2 : array-like, shape=[..., *point_shape]
            Point to evaluate.
        Returns
        -------
        belongs : array-like, shape=[..., ..., 1]
            Float evaluating the distance between two points in the metric space.
        """
