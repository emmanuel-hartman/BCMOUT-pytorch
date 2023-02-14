import torch
import numpy as np
from ConeSpace import ConeOverM
import abc
device = torch.device("cuda:0")


class MeasuresSpace:
    """Class for space of measures over a metric space M. Should allow comparison"""

    def __init__(self, M, delta):
        """
        Parameters
        ----------
        M : MetricSpace
            Space which the measures are over
        """
        self.M = M
        self.CoM = ConeOverM(M,delta)
        self.dim = M.dim+1
        self.shape = -1
        
    def random(self, n_samples=1, n_supports=1, maxWeight=1):
        """Create some random measures over a metric space M according to a uniform distribution.
        Parameters
        ----------
        """
        ls=[]
        for i in range(0,n_samples):
            supports = self.M.random(n_supports)
            masses = maxWeight*torch.rand(n_supports,1)
            point = torch.concat([masses,supports], dim=1)
            ls+=[point]
        
        return ls
                

    def belongs(self, points):
        """Checks whether or not two or more measures are over the same base space M"""
        belongs = torch.zeros((len(points)), dtype=torch.bool)
        for i in range(0,len(points)):
            point=points[i]
            belongs[i]=torch.all(self.CoM.belongs(point))
        return belongs
    
    def _pairwise_distance(self,point1,point2, steps, eps):
        a = point1[0,:].to(device) #measures of point1
        b = point2[0,:].to(device) #measures of point2
        u = point1[1:,:].to(device) #supports of point1
        v = point2[1:,:].to(device) #supports of point2
        m = a.size(dim=0) #number of supports of point 1
        n = b.size(dim=0) #number of supports of point 2
        Omega1 = self.CoM._energy(u,v).to(device)
        Omega = torch.zeros((Omega1.shape[0]+1,Omega1.shape[1]+1)).to(device)
        Omega[1:,1:]= Omega1
        rowsum = torch.sum(Omega,0)
        colsum = torch.sum(Omega,1)
        Omega[0,:] = torch.div(rowsum,torch.linalg.norm(rowsum, dim=0))
        Omega[:,0] = torch.div(colsum,torch.linalg.norm(colsum, dim=0))

        cost = torch.zeros([steps])

        P = Omega
        P[P<0] = 0
        Q = P #Initialize a semi-coupling

        for k in range(0,steps):
            P,Q = _contractionRowCol(P,Q,Omega,a,b,m,n)
            cost[k+1,:]=_cost(P,Q,Omega,a,b,m,n)
            ind = k+1
            if (cost[k+1]-cost[k,:])/cost[k+1] < eps:
                break
        dist = torch.sqrt(sum(a.cpu())+sum(b.cpu())-2*calcF(P,Q,Omega).cpu())
        return dist,cost,ind,P,Q
            
        
        

    def distance(self, point1, point2, steps, eps):
        
        distance = torch.zeros((len(point1),len(point2)))
        for i in range(0, distance.shape[0]):
            for j in range(0,distance.shape[1]):
                distance[i,j]=self._pairwise_distance(point1[i], point2[j], steps, eps)
        return distance
    
    def _cost(self, P, Q, Omega):
        c = torch.sum(P*Q*Omega)
        return c

    def _rowNormalize(P_new, a, n):
        RowNormP_new = torch.sqrt(torch.sum(P_new*P_new,dim=1)/a.transpose(0,1))
        RowNormMatrix = RowNormP_new.repeat([n,1]).transpose(0,1)
        P_newNormalized = P_new/RowNormMatrix
        return P_newNormalized
    def _colNormalize(Q_new, b, m):
        ColNormQ_new = torch.sqrt(torch.sum(Q_new*Q_new,dim=0)/b.transpose(0,1))
        ColNormMatrix = ColNormQ_new.repeat([m,1])
        Q_newNormalized = Q_new/RowNormalMatrix
        return Q_newNormalized

    def _contractionRowCol(P,Q,Omega,a,b,m,n):
        P = _rowNormalize(Q*Omega,a,n)
        Q = _colNormalize(P*Omega,b,m)
        return P,Q
    
    def _getAssignment(a,b,m,n,Omega,steps,eps):
        P = Omega
        P[P<0] = 0
        Q = P
        for j in range(0,steps):
            P,Q = _contractionRowCol(P,Q,Omega,a,b,m,n)
        return P,Q
