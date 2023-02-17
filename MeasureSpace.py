import torch
import numpy as np
from MetricSpace import MetricSpace 
from Metric import Metric
from ConeSpace import ConeOverM
import abc

class MeasureSpace(MetricSpace):
    """Class for space of measures over a metric space M. Should allow comparison"""

    def __init__(self, M, delta, use_cuda=False, **kwargs):
        """
        Parameters
        ----------
        M : MetricSpace
            Space which the measures are over
        """
        kwargs.setdefault("metric", WFRMetric(M,delta,use_cuda))
        self.M = M
        self.delta=delta
        self.CoM = ConeOverM(M,delta)
        self.shape = float('inf')
        super().__init__( float('inf') , **kwargs)
        
    def random(self, samples=1, n_supports=1, maxWeight=1):
        """Create some random measures over a metric space M according to a uniform distribution.
        Parameters
        ----------
        """
        ls=[]
        for i in range(0,samples):
            supports = self.M.random(samples = n_supports)
            masses = maxWeight*torch.rand(1,n_supports)
            point = torch.concat([masses,supports], dim=0)
            ls+=[point]        
        return ls
    
    def belongs(self, points):
        belongs = torch.zeros((len(points)), dtype=torch.bool)
        for i in range(0,len(points)):
            point=points[i]
            belongs[i]=torch.all(self.CoM.belongs(point))
        return belongs
    
    def dissimilarity(self,point1,max_steps=10000,eps=1e-5):
        return self._metric.distance(point1,max_steps=10000,eps=1e-5)

class WFRMetric(Metric):    
    def __init__(self,M,delta,use_cuda):
        self.M=M
        self.delta =delta
        self.CoM = ConeOverM(M,delta)
        self.solver=UOTKantorovichSolver(self.CoM, use_cuda)
        super().__init__()

    def distance(self,point1,point2=None,max_steps=10000,eps=1e-5):        
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
    
    

class UOTKantorovichSolver:
    def __init__(self,CoM, use_cuda):
        self.CoM = CoM
        self.device = torch.device("cuda:0" if use_cuda else "cpu")
        
    def _contractionRowCol(self,P,Q,Omega,a,b,m,n):
        P = self._rowNormalize(Q*Omega,a,n)
        Q = self._colNormalize(P*Omega,b,m)
        return P,Q  
    
    def _rowNormalize(self,Pnew,a,n):
        sums = torch.sum(Pnew*Pnew,dim=1)
        zeros = sums==0
        RowNormPnew = torch.sqrt(sums.reshape(1,-1)/a.reshape(1,-1))
        RowNormMatrix = RowNormPnew.repeat([n,1]).transpose(0,1)
        Pnew[zeros,:]=0
        RowNormMatrix[zeros,:]=1
        PnewNormalized = Pnew/RowNormMatrix
        return PnewNormalized
    
    def _colNormalize(self,Qnew,b,m):
        sums = torch.sum(Qnew*Qnew,dim=0)
        zeros = sums==0
        ColumnNormQnew = torch.sqrt(sums.reshape(1,-1)/b.reshape(1,-1))
        ColumnNormMatrix = ColumnNormQnew.repeat([m,1])
        Qnew[:,zeros]=0
        ColumnNormMatrix[:,zeros]=1
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
        
        cost=np.zeros((max_steps+1,1))
        for k in range(0,max_steps):
            P,Q=self._contractionRowCol(P,Q,Omega,a,b,m,n)
            cost[k+1,:]=self._calcF(P,Q,Omega).cpu()
            ind=k+1
            if (cost[k+1]-cost[k,:])/cost[k+1]<eps:
                break   
                
        dist=torch.sqrt(2*(self.CoM.delta**2)*(a.sum()+b.sum()-2*self._calcF(P,Q,Omega)))
        return dist
    