# -*- coding: utf-8 -*-
"""
Bird GAN
Source file
----------------------------
Author: Andrew Francey
----------------------------
Date: 20/08/24
"""

import os
import torch
import json
import pickle
import random
import numpy as np
import torch.nn as nn



def Data_normalize(data):
    '''
    Normalizes a dataset to between the values of 0 and 1.
    
    Parameters
    ----------
    data : listof(Numpy array)
        A list of data point values
    
    return
    ------
    normalized_data : listof (numpy array)
        A normalized data set between values 0 and 1.
    '''
    
    ## Converts the data to a numpy array to speed up calculations.
    data = np.array(data)
    max_val = np.max(data) # Determine the max value in the dataset.
    min_val = np.min(data) # Determine the min value in the dataset.
    
    ## Transpose all the values in the set to the domain [0,1].
    normalized_data = (data - min_val)/(max_val-min_val)
    
    ## Convert the array back to the list.
    normalized_data = list(normalized_data)
    
    return normalized_data


def ConvWidth(w, p, k, s):
   '''
   Computes the output size of a convolution layer. 

   Parameters
   ----------
   w : Int
       Input layers width.
   p : Int
       Padding applied.
   k : Int
       Kernal applied.
   s : Int
       Stride applied.

   Returns
   -------
   Int.
       The Width of the new layer.
   '''
   
   return 1 + (w + 2*p - (k -1) -1)//s
   
   
def ConvTransWidth(w, p, k, s):
    '''
    Computes the output size of a transpose convolution layer. 

    Parameters
    ----------
    w : Int
        Input layers width.
    p : Int
        Padding applied.
    k : Int
        Kernal applied.
    s : Int
        Stride applied.

    Returns
    -------
    Int.
        The Width of the new layer.
    '''
    
    return (w - 1)*s - 2*p + (k - 1) + 1


class Data():
    '''
    Data object for the real images to train the GAN to generate new data of 
    the real data in this data object
    
    Parameters
    ----------
    path : Str
        The location of the pickle file for the training images.
    device : Str
        The device to store the data on either the CPU or a GPU. Default is CPU.
    
    Attributes
    ----------
    dataset : listof (Numpy Array)
        A list of numpy arrays with the values of the images RGB values between 
        0 and 1.
    n_size : Int
        The number of images in the training set.
    res : Int
        The width and height of the images in the dataset.
    channels : Int
        The number of channels of the images in the dataset. (i.e. 1 for grey 
        scale and 3 for RGB images).
    unbatched_dataset : Torch tensor
        The unbatched dataset with the RGB values between 0 and 1.
    unbatched_labels : Torch tensor
        A torch tensor of values of 1 to represent the real value.
        
    Methods
    -------
    create_batches(n_batches, device, shuffle)
        Creates n_batches number of batches from the dataset and sends it to
        the device. If shuffle is true each batch will be randomly split up.
    '''
    
    def __init__(self, path, device=torch.device('cpu')):
        
        ## Open the pickle file containing the list of numpy arrays of the data.
        with open(path, 'rb') as f:
            dataset = pickle.load(f)
        f.close()
        
        ## Normalize the dataset to between 0 and 1.
        self.dataset = Data_normalize(dataset)
        
        self.n_size = len(self.dataset) # The number of images in the dataset.
        self.res = dataset[0].shape[0] # The width and height of the images.
        self.channels = dataset[0].shape[2] # The number of channels of the images.
        
        ## Converts to an array for quicker conversion to tensor.
        dataset = np.array(self.dataset)
        
        ## Transpose the data to match the input of the pytorch convolution layers.
        dataset = np.transpose(dataset,(0,3,1,2))
        self.unbatched_dataset = torch.tensor(dataset).to(device)
        
        ## Create a tensor for the labels for training.
        self.unbatched_labels = torch.ones(self.unbatched_dataset.shape[0]).to(device)
        
    
    def create_batches(self, n_batches, device=torch.device('cpu'), shuffle=True):
        '''
        Creates n_batches number of batches from the dataset and sends it to
        the device device. If shuffle is true each batch will be randomized.

        Parameters
        ----------
        n_batches : Int
            Number of batches to split the dataset into.
        device : Torch device, optional
            The device in which to store the data. The default is the Cpu.
        shuffle : Bool, optional
            Indicates if the data should be shuffled. The default is True.

        Returns
        -------
        batched_data : Torch Tensor
            Tensor containing the batches of the data.
        batched_labels : Torch Tensor
            Tensor containing the values of the labels 1.
        '''
        
        ## Determine the size of the batches
        batch_size = self.n_size//n_batches
        
        ## Copy the dataset so the randomizing doesn't effect the original.
        batchable_data = self.dataset.copy()
        
        ## Shuffle the data if shuffle is true.
        if shuffle:
            random.shuffle(batchable_data)
        
        ## Any items that don't fit into the last batch will be removed so all
        ##  batches will be the same size.
        for i in range(self.n_size % n_batches):
            batchable_data.pop(-1)
        
        ## Convert to tensor for quicker conversion to tensor.
        batchable_data = np.array(batchable_data)
        batchable_data = torch.tensor(batchable_data)
        
        ## Reshape the tensor to include the batches.
        batched_data = batchable_data.reshape((n_batches, batch_size, 
                                               self.res,self.res,self.channels))
        batched_labels = torch.ones((n_batches, batch_size))
        
        ## Send to the device to store the data.
        batched_data.to(device)
        batched_labels.to(device)
        
        return batched_data, batched_labels
    

class Generator(nn.Module):
    '''
    A 5 layer 2D transposed convolution network for generating images from a
    randomized latent vector.
    
    Attrabutes
    ----------
    n_lat : Int
        The size of the latent vector.
    main : Torch Sequential
        The layers and activations functions constructing the network.
    
    Methods
    -------
    forward(x)
        Runs the vector x through the network converting the vector into a 
        tensor representing an image.
    '''
    
    def __init__(self, data, param):
        super(Generator, self).__init__()
        
        ## Extract the values for the network from the param file.
        k1 = param['kernal_1']
        p1 = param['padding_1']
        s1 = param['stride_1']
        
        k2 = param['kernal_2']
        p2 = param['padding_2']
        s2 = param['stride_2']
        
        k3 = param['kernal_3']
        p3 = param['padding_3']
        s3 = param['stride_3']
        
        k4 = param['kernal_4']
        p4 = param['padding_4']
        s4 = param['stride_4']
        
        k5 = param['kernal_5']
        p5 = param['padding_5']
        s5 = param['stride_5']
        
        Conv1 = param['Conv1']
        Conv2 = param['Conv2']
        Conv3 = param['Conv3']
        Conv4 = param['Conv4']
        
        ## Solve for the initial latent space vectors size.
        width4 = ConvWidth(data.res, p5, k5, s5)
        width3 = ConvWidth(width4, p4, k4, s4)
        width2 = ConvWidth(width3, p3, k3, s3)
        width1 = ConvWidth(width2, p2, k2, s2)
        self.n_lat = ConvWidth(width1, p1, k1, s1)
        
        ## Construct the network.
        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.n_lat, Conv1, k1, s1, p1, bias=False),
            nn.BatchNorm2d(Conv1),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(Conv1, Conv2, k2, s2, p2, bias=False),
            nn.BatchNorm2d(Conv2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(Conv2, Conv3, k3, s3, p3, bias=False),
            nn.BatchNorm2d(Conv3),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(Conv3, Conv4, k4, s4, p4, bias=False),
            nn.BatchNorm2d(Conv4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(Conv4, data.channels, k5, s5, p5, bias=False),
            nn.Sigmoid()
            )
    
    def forward(self, x):
        '''
        Runs a vector x through the generator network to create an image.

        Parameters
        ----------
        x : Torch Tensor
            The randomized laten space vector.

        Returns
        -------
        Torch Tensor
            The tensor values for the generated image.
        '''
        
        return self.main(x)

        
class Discriminator(nn.Module):
    '''
    A 5 layer convolation with a single fully connected layer producing the
    confidince in determining if the feed in image is fake or real.
    
    Attributes
    ----------
    main : Torch Sequential
        The layers and activations functions constructing the network.
    fc : Torch Sequential
        The fully connected layer of the network.
        
    Methods
    -------
    forward(x)
        Runs the vector x through the network producing a confidence value
        between 0 and 1 in which the image is real (i.e. 1) or fake (i.e. 0).
    '''
    def __init__(self, data, param):
        super(Discriminator, self).__init__()
        
        ## Extract the values for the network from the param file.
        k1 = param['kernal_1']
        p1 = param['padding_1']
        s1 = param['stride_1']
        
        k2 = param['kernal_2']
        p2 = param['padding_2']
        s2 = param['stride_2']
        
        k3 = param['kernal_3']
        p3 = param['padding_3']
        s3 = param['stride_3']
        
        k4 = param['kernal_4']
        p4 = param['padding_4']
        s4 = param['stride_4']
        
        k5 = param['kernal_5']
        p5 = param['padding_5']
        s5 = param['stride_5']
        
        Conv1 = param['Conv1']
        Conv2 = param['Conv2']
        Conv3 = param['Conv3']
        Conv4 = param['Conv4']
        Conv5 = param['Conv5']
        
        ## Solve for the size of the last convolation layer.
        width1 = ConvWidth(data.res, p1, k1, s1)
        width2 = ConvWidth(width1, p2, k2, s2)
        width3 = ConvWidth(width2, p3, k3, s3)
        width4 = ConvWidth(width3, p4, k4, s4)
        width5 = ConvWidth(width4, p5, k5, s5)
        
        ## The main convolation layers.
        self.main = nn.Sequential(
            nn.Conv2d(data.channels, Conv1, k1, s1, p1, bias = False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(Conv1, Conv2, k2, s2, p2, bias=False),
            nn.BatchNorm2d(Conv2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(Conv2, Conv3, k3, s3, p3, bias=False),
            nn.BatchNorm2d(Conv3),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(Conv3, Conv4, k4, s4, p4, bias=False),
            nn.BatchNorm2d(Conv4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(Conv4, Conv5, k5, s5, p5, bias=False),
            nn.BatchNorm2d(Conv5),
            nn.LeakyReLU(0.2, inplace=True),
            )
        
        # The fully connected layer.
        self.fc = nn.Sequential(
            nn.Linear((width5**2)*Conv5, 1),
            nn.Sigmoid()
            )
        
    def forward(self, x):
        '''
        Runs a tensor representation of a image through the networks layers 
        producing a confidence value in if the image is fake or real.

        Parameters
        ----------
        x : Torch tensor
            A tensor representation of an image to determine if its real or 
            fake.

        Returns
        -------
        x : Torch Tensor
            A value representing the networks confidence in if the image is
            real (i.e. 1) or fake (i.e. 0).
        '''
        x = self.main(x)
        x = x.reshape(x.size(0),-1) # Flattens the convolation into a column vector.
        x = self.fc(x)
        
        return x
    
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        