import abc
import torch

class LengthMetric(abc.ABC):    
    """Class for an abstract Metric space
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    @abc.abstractmethod
    def geodesic(self, point1, point2, t):
        """Compute t equally-spaced points on the distance-minimizing path between two points.
        Parameters
        ----------
        point1 : array-like, shape=[point_shape]
            Point to evaluate.
        point2 : array-like, shape=[point_shape]
            Point to evaluate.
        Returns
        -------
        geodesic : array-like, shape=[t, point_shape]
            Length minimizing path between two points in the metric space.
        """        
