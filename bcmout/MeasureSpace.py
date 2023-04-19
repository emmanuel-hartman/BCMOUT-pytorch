import abc
import torch
from bcmout.MetricSpace import MetricSpace 
from bcmout.Metric import Metric
from bcmout.ConeSpace import ConeOverM
from bcmout.BCMSolver import BCMSolver

class MeasureSpace(MetricSpace):
    """ Class for the metric space defined by the cone over a metric space
    
    Parameters
    ----------
    M : Metric Space Object
        metric space that we are defining the space of measures over.
    delta : float64 
        parameter that defines the metric on the space of measures .
    metric : string (Default: None) 
        String specifying what type of metric to equip the space of measures with        
    """

    def __init__(self, M, delta, metric=None, use_cuda=False,**kwargs):
        
        if metric == "WFR":
            self.CoM = ConeOverM(M,delta,metric="CosBar")
            kwargs.setdefault("metric", WFRMetric(M,delta,use_cuda))
        elif metric=="GH":
            self.CoM = ConeOverM(M,delta,metric="Exp")
            kwargs.setdefault("metric", GHMetric(M,delta,use_cuda))
        else:
            self.CoM = ConeOverM(M,delta,metric="CosBar")
            kwargs.setdefault("metric", WFRMetric(M,delta,use_cuda))
            
        self.M = M
        self.delta=delta
        self.shape = float('inf')
        super().__init__( float('inf') , **kwargs)
        
    def random(self, samples=1, n_supports=1, bound=1.0):
        """Sample random points on the metric space according to a uniform distribution.
        Parameters
        ----------
        samples : int
            Number of samples.
            Optional, default: 1
        n_supports : int
            Number of supports of the measure on M.
            Optional, default: 1
        bound : float64
            Bound for the measure masses we sample from 
            Optional, default: 1.0
        Returns
        -------
        points : list-like, shape=[n_samples]
            Points sampled in the metric space.
        """
        ls=[]
        for i in range(0,samples):
            supports = self.M.random(samples = n_supports)
            masses = bound*torch.rand(1,n_supports)
            point = torch.concat([masses,supports], dim=0)
            ls+=[point]        
        return ls
    
    def belongs(self, points):
        """Evaluate if a point belongs to the metric space.
        Parameters
        ----------
        points : array-like, shape=[point_shape,num_points]
            Point to evaluate.
        Returns
        -------
        belongs : array-like, shape=[num_points]
            Boolean evaluating if point belongs to the metric space.
        """
        belongs = torch.zeros((len(points)), dtype=torch.bool)
        for i in range(0,len(points)):
            point=points[i]
            belongs[i]=torch.all(self.CoM.belongs(point))
        return belongs
    
    def dissimilarity(self,point1,max_steps=10000,eps=1e-5):
        return self._metric.distance(point1,max_steps=max_steps,eps=1e-5)

class WFRMetric(Metric):    
    """Class for a Wasserstein-Fisher-Rao metric object.
    
    Parameters
    ----------
    M : Metric Space Object
        metric space that we are defining the measure space over.
    delta : float64 
        parameter that defines the metric on the cone.
    use_cuda : bool 
        parameter that defines whether to use cuda.
    """
    def __init__(self,M,delta,use_cuda):
        self.M=M
        self.delta =delta
        self.CoM = ConeOverM(M,delta)
        self.solver=BCMSolver(use_cuda)
        super().__init__()

    def distance(self,point1,point2=None,max_steps=10000,eps=1e-5):  
        """Compute the distance between two points.
        Parameters
        ----------
        point1 : list-like, shape=[num_points1]
            Point to evaluate.
        point2 : list-like, shape=[num_points2]
            Point to evaluate.
        Returns
        -------
        belongs : array-like, shape=[num_points1, num_points2]
            Float evaluating the distance between two points in the metric space.
        """  
        if point2 is not None:
            distance = torch.zeros((len(point1),len(point2)))
            for i in range(0, distance.shape[0]):
                for j in range(0,distance.shape[1]):
                    print(str(i*distance.shape[0]+j+1)+"/"+str(distance.shape[0]*distance.shape[1]), end="\r")
                    a = point1[i][0,:]
                    b = point2[j][0,:]
                    u = point1[i][1:,:]
                    v = point2[j][1:,:]
                    Omega = self.CoM._energy(u,v)
                    P,Q = self.solver.getOptimalPQ(a,b,Omega,max_steps,eps)
                    distance[i,j]=self.solver.getDistFromPQ(P,Q,a,b,Omega,self.CoM.delta)
        else:
            distance = torch.zeros((len(point1),len(point1)))
            for i in range(0, distance.shape[0]):
                for j in range(i+1,distance.shape[1]):
                    print(str(i*distance.shape[0]+j+1)+"/"+str(distance.shape[0]*distance.shape[1]-distance.shape[0]), end="\r")
                    a = point1[i][0,:]
                    b = point1[j][0,:]
                    u = point1[i][1:,:]
                    v = point1[j][1:,:]
                    Omega = self.CoM._energy(u,v)
                    P,Q = self.solver.getOptimalPQ(a,b,Omega,max_steps,eps)
                    distance[i,j]=self.solver.getDistFromPQ(P,Q,a,b,Omega,self.CoM.delta)
                    distance[j,i]=distance[i,j]
        return distance
    
class GHMetric(Metric):    
    """Class for a Gaussian-Hellinger metric object.
    
    Parameters
    ----------
    M : Metric Space Object
        metric space that we are defining the measure space over.
    delta : float64 
        parameter that defines the metric on the cone.
    use_cuda : bool 
        parameter that defines whether to use cuda.
    """    
    def __init__(self,M,delta,use_cuda):
        self.M=M
        self.delta =delta
        self.CoM = ConeOverM(M,delta,metric="Exp")
        self.solver=BCMSolver(use_cuda)
        super().__init__()

    def distance(self,point1,point2=None,max_steps=10000,eps=1e-5):  
        """Compute the distance between two points.
        Parameters
        ----------
        point1 : list-like, shape=[num_points1]
            Point to evaluate.
        point2 : list-like, shape=[num_points2]
            Point to evaluate.
        Returns
        -------
        belongs : array-like, shape=[num_points1, num_points2]
            Float evaluating the distance between two points in the metric space.
        """        
        if point2 is not None:
            distance = torch.zeros((len(point1),len(point2)))
            for i in range(0, distance.shape[0]):
                for j in range(0,distance.shape[1]):
                    print(str(i*distance.shape[0]+j+1)+"/"+str(distance.shape[0]*distance.shape[1]), end="\r")
                    a = point1[i][0,:]
                    b = point2[j][0,:]
                    u = point1[i][1:,:]
                    v = point2[j][1:,:]
                    Omega = self.CoM._energy(u,v)
                    P,Q = self.solver.getOptimalPQ(a,b,Omega,max_steps,eps)
        else:
            distance = torch.zeros((len(point1),len(point1)))
            for i in range(0, distance.shape[0]):
                for j in range(i+1,distance.shape[1]):
                    print(str(i*distance.shape[0]+j+1)+"/"+str(distance.shape[0]*distance.shape[1]-distance.shape[0]), end="\r")
                    a = point1[i][0,:]
                    b = point1[j][0,:]
                    u = point1[i][1:,:]
                    v = point1[j][1:,:]
                    Omega = self.CoM._energy(u,v)
                    P,Q = self.solver.getOptimalPQ(a,b,Omega,max_steps,eps)
                    distance[j,i]=distance[i,j]
        return distance