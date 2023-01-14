import torch
import numpy as np
import ConeOverM.py
import abc
device = torch.device("cuda:0")


class MeasuresSpace:
    """Class for space of measures over a metric space M. Should allow comparison"""

    def __init__(self):
        """
        Parameters
        ----------
        M : MetricSpace
            Space which the measures are over
        """
        return

    def distance(self):
        """Here we have our algorithm"""
        
        return

    def random(self, n_samples, points=0, n_points=0):
        """Create some random measures over a metric space M according to a uniform distribution.
        Parameters
        ----------
        points : array-like
            array of points in M
        n_samples : int
            number of samples to be taken
        n_points : int
            number of points to be randomly sampled. 
        """
        if points != 0:
            
        elif n_points !=0:

        else:
            raise NameError('Please provide an array of points or a number of points to be sampled')
                

    def belongs(self, Measures):
        """Checks whether or not two or more measures are over the same base space M"""
        for i in Measures:
        
        return belongs
    
    
