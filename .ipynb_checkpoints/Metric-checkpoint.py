import abc
import torch

class Metric(abc.ABC):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    @abc.abstractmethod
    def distance(self, point1, point2):
        """Compute the distance matrix between two sets of points.
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