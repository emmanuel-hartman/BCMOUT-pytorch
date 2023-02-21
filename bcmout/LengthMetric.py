import abc
import torch
from bcmout.Metric import Metric

class LengthMetric(Metric):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    @abc.abstractmethod
    def geodesic(self, point1, point2, t):
        """Compute the distance matrix between two points.
        Parameters
        ----------
        point1 : array-like, shape=[*point_shape]
            Point to evaluate.
        point2 : array-like, shape=[*point_shape]
            Point to evaluate.
        Returns
        -------
        geodesic : array-like, shape=[t, *point_shape]
            Length minimizing path between two points in the metric space.
        """        
