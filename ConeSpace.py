import abc
import torch
import numpy as np
device = torch.device("cuda:0")

class ConeOverM(MetricSpace):
    """
    Parameters
    ----------
    dim : int
        in case we're able to speak of the dimension of the cone space
    shape : int
        shape of the measure space + 1
    """
    def __init__(self, Measures, delta):
        self.shape = MeasuresSpace.M.shape + 1
        self.points = 
        return
    
    def belongs(self,point,atol):        
        return 

    def randomPoint(self, samples=1):
        for i in samples:

            return


    def distance(self, points):
        """Takes an array of points, separates them into the metric part and the measure part, and computes the cone distance between them
        Parameters
        ----------
        points : array-like, shape = [..., points]

        
        
        Returns
        -------
        distance : matrix like, shape = [..., points, points]"""
        
        d = np.zeros(

        return distance
