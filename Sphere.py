import torch
device = torch.device("cuda:0")


class Hypersphere(MetricSpace):
    """Class for a hypersphere of radius 1 centered at the origin
    """
    
    def __init__(self, dim, shape):
        self.dim = dim
        self.shape = shape
        super().__init__(shape)
      
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
        rand = array-like
            Random numbers to be used in the projection onto the hypersphere
        points = array-like
            Points on the sphere following projected from 
        Returns
        -------
        samples : array-like, shape=[..., *point_shape]
            Points sampled in the metric space.
        """
        rand = torch.rand(n_samples, shape)
        points = torch.zeros(n_samples, shape)
        for i in n_samples:
            points[i,:] = rand[i,:]/torch.linalg.norm(rand[i,:])            
        
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
