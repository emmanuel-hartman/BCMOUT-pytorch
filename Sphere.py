import torch
device = torch.device("cuda:0")


class Sphere(MetricSpace):
    """
    """
    
    def __init__(self, dim):
        self.dim = dim
        shape=dim+1
        super().__init__(dim=dim)
      
    def belongs(self, point, atol):
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
        
        return torch.isclose(torch.linalg.norm(point,dim=0), torch.ones((point.shape[0])), atol=atol)
        
        
    def random_point(self, n_samples=1):
        """Sample random points on the metric space according to a uniform distribution.
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
        return 
    
    def distance(self,point1,point2):
        """Compute the distance between two points.
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
        
        return torch.acos(torch.einsum('ai,bi->ab', point1,point2))
