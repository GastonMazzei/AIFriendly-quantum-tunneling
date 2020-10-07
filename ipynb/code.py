#!/bin/env/python

from functools import partial

import numpy as np
import pandas as pd

def square_potential(self,x):
  if (x>self.center+self.width/2 
      or 
      x<self.center-self.width/2): return self.offset 
  else:
    output = self.offset - self.depth
    return output

def triangle_potential(self,x):
  if (x>self.center+self.width/2 
     or 
     x<self.center-self.width/2): return self.offset 
  else:
    output = self.offset 
    output -= (1-abs(x-self.center)/(self.width/2)) * self.depth
    return output

class symmetric_well:
  def __init__(self):
    mssg =     """
    INSTANTIATED:
    Potential well that 
    is built with 
    self.build(width=1, 
              depth=np.inf,
              center=0,
              offset=0,
              shape=square)
    available shapes:
    square, triangle
    """
    print(mssg)

  def build(self, **kwargs):
    self.width = kwargs.get('width',1)
    self.depth = kwargs.get('depth',np.inf)
    self.center = kwargs.get('center',0)
    self.offset = kwargs.get('offset',0)
    self.shape = kwargs.get('shape','square')
    if self.shape=='square':
      self.potential = partial(square_potential,self) 
    elif self.shape=='triangle':
      self.potential = partial(triangle_potential,self)
    return 

if __name__=='__main__':
  well = symmetric_well()
  well.build(shape='triangle')
  import matplotlib.pyplot as plt
  plt.plot(well.potential(np.asarray([0,1,2,3,4,5])))