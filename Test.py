import torch
import numpy as np
from MeasuresSpace import MeasuresSpace
from Sphere import Sphere
device = torch.device("cuda:0")
M = Sphere(3)
Measures = MeasuresSpace(M, 2)
k1 = Measures.random(1,3,4)
k2 = Measures.random(1,4,5)
m = Measures.distance(k1,k2, 100, 0.001)
print(m)
#Omega1 = Measures.CoM._energy(u,v)

#R = torch.eye(3).to(device)
#Omega2 = torch.einsum('kl,ik,jl->ij',R,u,v)

#print(Omega1)
#print(Omega2)

#Omega = torch.zeros((Omega1.shape[0]+1,Omega1.shape[1]+1)).to(device)
#Omega[1:,1:]= Omega1
#rowsum = torch.sum(Omega,0)
#colsum = torch.sum(Omega,1)
#Omega[0,:] = torch.div(rowsum,rowsum)
#Omega[:,0] = torch.div(colsum,colsum)
#Omega[0,0] = 0
#P = Omega
#P[P<0] = 0
#Q = P

