# -*- coding: utf-8 -*-
"""
Bird GAN
Train file
-----------------------------------
Created on Mon Aug 26 00:48:06
-----------------------------------
@author: Andrew Francey
"""


import os
import torch
import json, argparse
import pickle
import random
import winsound
import time
import source as src
import torch.nn as nn
import torch.optim as optim


##-----------------------------------------------------------------------------
## Functions
##-----------------------------------------------------------------------------

def prep(param, device):
    '''
    Consumes a JSON file with hyperparameters for the generator and 
    discriminator. Processes the data into a more workable formate. Creates
    the generator and the discriminator networks to be trained.

    Parameters
    ----------
    param : JSON file
        Hyperparameters for the networks.
    device : Torch device
        Device to store the models and data on.

    Returns
    -------
    data : Data object
        The processed data for training of the GAN.
    gen : Net object
        The generator network to generate images.
    disc : Net object
        The discriminator for deciding if an image is real or fake.
    '''
    
    data = src.Data(args.data_path, device)
    gen = src.Generator(data, param['Generator']).to(device)
    disc = src.Discriminator(data, param['Discriminator']).to(device)
    
    return data, gen, disc


def run(param, gen, disc, data, device):
    
    ## Using the bineary cross entropy loss function to determine the loss values
    loss = nn.BCELoss()
    
    ## Using the ADAM optimizer to optimize both networks.
    optimizerD = optim.Adam(disc.parameters(), lr=param['learn_rate'], 
                            batas=(param['beta1'],param['beta2']))
    optimizerG = optim.Adam(gen.parameters(), lr=param['learn_rate'],
                            bates=(param['beta1'],param['beta2']))
    
    loss_vals = []
    cross_vals = []
    img_list = []
    
    fake_labels = torch.zeros((data.batch_size), float).to(device)
    
    num_epochs = int(param['num_epochs'])
    for epoch in range(num_epochs):
        
        n_batches = param['num_batches']
        batches, real_labels = data.create_batches(n_batches, device, args.shuffle)
            
        for i, batch in enumerate(batches, 0):
            
            ## Train the discriminator
            ## --------------------------------------
            disc.zero_grad()
            
            ## Train with all-real batch
            output = disc.forward(batch)
            
            ## Calculate loss on all-real batch
            lossD_real = loss(output, real_labels)
            
            ## Calculate the gradient on a backward pass
            lossD_real.backward()
            
            ## Train with all-fake batch
            
            ## Create batchs of the latent vectors
            noise = torch.randn(data.batch_size, gen.n_lat, 1, 1, device=device)
            
            ## Generate fake images
            fake = gen.forward(noise)
            
            ## Forward pass the fake images through the discriminator
            output = disc.forward(fake)
            
            ## Calculate the loss on all-fake batch
            lossD_fake = loss(output, fake_labels)
            
            ## Calculate the gradient on this batch, accumulated with the above
            lossD_fake.backward()
            
            ## Calculate the error of the discriminator from both the real and fake.
            lossD = lossD_real + lossD_fake
            
            ## Update the discriminator
            optimizerD.step()
            
            
            ## Train the generator
            ## -------------------------------------
            gen.zero_grad()
            

if __name__ == "__main__":
    
    start_time = time.time() # Get the start time.
    
    ## Determine if there is a GPU to use for training.
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    
    device = torch.device(dev)
    
    ## Determine the root path for the files.
    path = os.path.dirname(__file__) + '\\'
    
    ## Create arguments that are needed for the training. This arguments can be
    ##  changed from the defaults in the command line.
    parser = argparse.ArgumentParser(description='Training of a GAN')
    parser.add_argument('--param', default=path+'param.json',type=str,
                        help='File path for the Json file with the hyperparameters')
    parser.add_argument('--data-path', default=path+'\\Dataset\\64images.pickle',
                        type=str, help='Location of the training images')
    parser.add_argument('--model-name', default=path+'64BirdGAN.pkl',
                        type=str, help='Name to save the model as.')
    parser.add_argument('--batch', default='True',type=bool,
                        help='Boolean value to use batching.')
    args = parser.parse_args()
    
    ## Open the hyperparameter file.
    with open(args.param) as paramfile:
        param = json.load(paramfile)
    paramfile.close()
    
    data, gen, disc = prep(param, device)