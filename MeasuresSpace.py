import torch
import numpy as np
import ConeOverM.py
import abc
device = torch.device("cuda:0")


class MeasuresSpace:
    """Class for space of measures. Should allow comparison """

    def __init__(self):
        """
        Parameters
        ----------
        """
        return

    def distance(self):
        """Here we have our algorithm"""
        
        return

    def random(self, M, samples=1):
        """Create some random measures over M according to a uniform distribution
        Parameters
        ----------
        """
        Measures = np.zeros(samples)
        for i in range(samples):
            for i in range (
            
            return Measures
         
        return Measure1

    def belongs(self, Measures):
        """Checks whether or not two or more measures are over the same base space M"""
        for i in 
        
        return belongs
    
    
