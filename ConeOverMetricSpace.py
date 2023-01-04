import abc
import torch
device = torch.device("cuda:0")

class ConeOverM(MetricSpace):
    """
    Parameters
    ----------
    dim : int
        Dimension of the cone space (dimension of the manifold plus one). 
    shape : tuple of int
        Shape of one element of M 
    """
    def __init__(self,metricSpace, measure, dim, ):
        self.dim = metricSpace.dim + 1
        self.point_ndim = len(self.shape) + 1
        return
    
    def belongs(self,point,atol):

        return torch.isclose(torch.linalg.norm(point,dim=0), torch.ones((point.shape[0])), atol=atol)

    def randomPoint(self, samples=1):
        for i in samples:

            return


    def distance(self, point1, point2):
        

        return
