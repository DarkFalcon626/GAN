# -*- coding: utf-8 -*-
"""
GAN Viewer and Generator
--------------------------------
Created on Fri Sept 06 02:40:00
--------------------------------
@author: Andrew Francey
"""

import os
import torch
import pickle
import torchvision.transforms as trans
from PIL import Image


##------------------------------------------
## Functions
##------------------------------------------


def View(image):
    '''
    Consumes a torch tensor representing an image and shows the image the 
    tensor represents.

    Parameters
    ----------
    image : Torch tensor
        Tensor of an image.

    Effects
    -------
    Shows the image of the tensor on screen.
    
    Returns
    -------
    None.
    '''
    
    transform = trans.ToPILImage()
    
    img = transform(image)
    img.show()
    
    
        
    