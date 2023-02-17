import torch
import scipy.linalg
import numpy as np
from MetricSpace import MetricSpace 
from Metric import Metric 
device = torch.device("cuda:0")


class SPDMatrices(MetricSpace):
    """Class for SPDMatrices
    """
    
    def __init__(self, n,**kwargs):
        kwargs.setdefault("metric", WassersteinBuresMetric(n))
        self.dim = int((n*(n+1))/2)
        self.shape= n*n
        self.n = n
        super().__init__(int((n*(n+1))/2),**kwargs)
      
    def belongs(self, points, atol=1e-6):
        """Evaluate if a point belongs to the metric space.
        Parameters
        ----------
        point : array-like, shape=[num_points,point_shape]
            Point to evaluate.
        atol : float
            Absolute tolerance.
        Returns
        -------
        belongs : array-like, shape=[num_points]
            Boolean evaluating if point belongs to the metric space.
        """
        points = points.reshape(self.n,self.n,-1)
        is_sy=torch.abs(points-points.transpose(0,1)).sum(dim=(0,1))
        is_sy=is_sy<1e-05
        is_pd=torch.tensor([self._is_mat_pd(points[:,:,i]) for i in range(0,points.shape[2])],dtype=torch.bool)
        return  torch.logical_and(is_sy, is_pd)
        
        
    def random(self, samples=1, bound=1.0):
        """Sample random points on the metric space according to a uniform distribution.
        Parameters
        ----------
        n_samples : int
            Number of samples.
            Optional, default: 1.
        Returns
        -------
        samples : array-like, shape=[point_shape,n_samples]
            Points sampled in the metric space.
        """
        points = bound * (2 * torch.rand((self.n,self.n,samples)) - 1)
        points = .5*(points+points.transpose(0,1))
        points = torch.linalg.matrix_exp(points.transpose(0,2)).transpose(0,2)
        return points.reshape((-1,samples))
    
    def _is_mat_pd(self,x):
        try:
            torch.linalg.cholesky(x)
            return True
        except RuntimeError:
            return False

    
class WassersteinBuresMetric(Metric):
    def __init__(self,n,):
        self.n=n
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
        
        point1=point1.reshape((self.n,self.n,point1.shape[1],1))
        point2=point2.reshape((self.n,self.n,1,point2.shape[1]))
        point1=point1.transpose(0,2).transpose(1,3)
        point2=point2.transpose(0,2).transpose(1,3)
        mul = torch.matmul(point1, point2)
        sqrt = self.sqrtm(mul)
        tr1 = self._trace(point1)
        tr2 = self._trace(point2)
        trsqrt = self._trace(sqrt)
        sqddist = tr1+tr2-2*trsqrt
        sqddist[sqddist<0]=0
        return torch.sqrt(.5*(sqddist+sqddist.transpose(0,1)))
    
    def sqrtm(self,x):
        np_sqrtm = np.vectorize(scipy.linalg.sqrtm, signature="(n,m)->(n,m)")(x)
        return torch.from_numpy(np_sqrtm.real)
        
    def _trace(self,x):
        return x.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
