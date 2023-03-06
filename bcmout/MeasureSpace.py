import abc
import torch
from bcmout.MetricSpace import MetricSpace 
from bcmout.Metric import Metric
from bcmout.ConeSpace import ConeOverM

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
            kwargs.setdefault("metric", CosBarMetric(M,delta,use_cuda))
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
        self.solver=BCDSolver(self.CoM, use_cuda)
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
                    distance[i,j]=self.solver._pairwise_distance(point1[i], point2[j],max_steps,eps)
                    distance[j,i]=distance[i,j]
        else:
            distance = torch.zeros((len(point1),len(point1)))
            for i in range(0, distance.shape[0]):
                for j in range(i+1,distance.shape[1]):
                    print(str(i*distance.shape[0]+j+1)+"/"+str(distance.shape[0]*distance.shape[1]-distance.shape[0]), end="\r")
                    distance[i,j]=self.solver._pairwise_distance(point1[i], point1[j],max_steps,eps)
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
        self.solver=BCDSolver(self.CoM, use_cuda)
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
                    distance[i,j]=self.solver._pairwise_distance(point1[i], point2[j],max_steps,eps)
                    distance[j,i]=distance[i,j]
        else:
            distance = torch.zeros((len(point1),len(point1)))
            for i in range(0, distance.shape[0]):
                for j in range(i+1,distance.shape[1]):
                    print(str(i*distance.shape[0]+j+1)+"/"+str(distance.shape[0]*distance.shape[1]-distance.shape[0]), end="\r")
                    distance[i,j]=self.solver._pairwise_distance(point1[i], point1[j],max_steps,eps)
                    distance[j,i]=distance[i,j]
        return distance
    
    

class BCDSolver: 
    """Class for a BCD solver for Kantorovich formulations of OUT.
    
    Parameters
    ----------
    CoM : Metric Space Object
        cone space object used for the initialization of the BCD solver
    use_cuda : bool 
        parameter that defines whether to use cuda.
    """ 
    def __init__(self,CoM, use_cuda):
        self.CoM = CoM
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        
    def _contractionRowCol(self,P,Q,Omega,a,b,m,n):
        P = self._rowNormalize(Q*Omega,a,n)
        Q = self._colNormalize(P*Omega,b,m)
        return P,Q  
    
    def _rowNormalize(self,Pnew,a,n):
        sums = torch.sum(Pnew*Pnew,dim=0)
        zeros = sums==0
        RowNormPnew = torch.sqrt(sums.reshape(1,-1)/a.reshape(1,-1))
        RowNormMatrix = RowNormPnew.repeat([n,1])
        
        Pnew[:,zeros]=0
        RowNormMatrix[:,zeros]=1
        PnewNormalized = Pnew/RowNormMatrix
        return PnewNormalized
    
    def _colNormalize(self,Qnew,b,m):
        sums = torch.sum(Qnew*Qnew,dim=1)
        zeros = sums==0
        ColumnNormQnew = torch.sqrt(sums.reshape(1,-1)/b.reshape(1,-1))
        ColumnNormMatrix = ColumnNormQnew.repeat([m,1]).transpose(0,1)
        Qnew[zeros,:]=0
        ColumnNormMatrix[zeros,:]=1
        QnewNormalized = Qnew/ColumnNormMatrix
        return QnewNormalized
    
    def _calcF(self,P,Q,Omega):
        cost=torch.sum(P*Q*Omega)
        return cost

    
    def _pairwise_distance(self,point1,point2,max_steps,eps):
        point1=point1.to(self.device)
        point2=point2.to(self.device)
        a = point1[0,:]
        b = point2[0,:]
        m = a.shape[0]
        n = b.shape[0]
        u = point1[1:,:]
        v = point2[1:,:]
        Omega = self.CoM._energy(u,v)
        P = Omega
        Q = Omega
        
        cost=torch.zeros((max_steps+1,1))
        for k in range(0,max_steps):
            P,Q=self._contractionRowCol(P,Q,Omega,a,b,m,n)
            cost[k+1,:]=self._calcF(P,Q,Omega).cpu()
            ind=k+1
            if (cost[k+1]-cost[k,:])/cost[k+1]<eps:
                break   
                
        dist=2*(self.CoM.delta)*torch.sqrt((a.sum()+b.sum()-2*self._calcF(P,Q,Omega)))
        return dist