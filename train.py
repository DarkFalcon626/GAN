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
                            betas=(param['beta1'],param['beta2']))
    optimizerG = optim.Adam(gen.parameters(), lr=param['learn_rate'],
                            betas=(param['beta1'],param['beta2']))
    
    loss_vals = []
    img_list = []
    
    num_epochs = int(param['num_epochs'])
    for epoch in range(num_epochs):
        
        n_batches = param['num_batches']
        batches, real_labels = data.create_batches(n_batches, device, args.shuffle)

        fake_labels = torch.zeros(data.batch_size, dtype=torch.float64).to(device)
        
        batch_errorD = 0
        batch_errorG = 0
        
        for i, batch in enumerate(batches, 0):
            
            ## Train the discriminator
            ## --------------------------------------
            disc.zero_grad()
            
            ## Train with all-real batch
            output = disc.forward(batch).view(-1)
            
            ## Calculate loss on all-real batch
            lossD_real = loss(output, real_labels)
            
            ## Calculate the gradient on a backward pass
            lossD_real.backward(retain_graph=True)
            
            ## Train with all-fake batch
            
            ## Create batchs of the latent vectors
            noise = torch.randn(data.batch_size, gen.n_lat, 1, 1,
                                dtype=torch.float64, device=device)
            
            ## Generate fake images
            fake = gen.forward(noise)
            
            ## Forward pass the fake images through the discriminator
            output = disc.forward(fake).view(-1)
            
            ## Calculate the loss on all-fake batch
            lossD_fake = loss(output, fake_labels)
            
            ## Calculate the gradient on this batch, accumulated with the above
            lossD_fake.backward(retain_graph=True)
            
            ## Calculate the error of the discriminator from both the real and fake.
            lossD = lossD_real + lossD_fake
            
            ## Update the discriminator
            optimizerD.step()
            
            
            ## Train the generator
            ## -------------------------------------
            gen.zero_grad()
            
            ## Feed the fake images through the discriminator that was just
            ## updated.
            output = disc.forward(fake).view(-1)
            
            ## Calculate loss, fake labels are real for generator cost.
            lossG = loss(output, real_labels)
            
            ## Calculate the gradients for G
            lossG.backward(retain_graph=True)
            
            ## Update G.
            optimizerG.step()
            
            ## Add the errors to the batches errors.
            batch_errorD += lossD
            batch_errorG += lossG
        
        ## Determine the average error of the batches.
        lossD = batch_errorD/n_batches
        lossG = batch_errorG/n_batches
        
        loss_vals.append([lossG,lossD])
        
        ## Generate an image to visually test how the generator is preforming.
        test_img_noise = torch.randn(1,gen.n_lat,1,1, dtype=torch.float64,
                                     device=device)
        test_img = gen.forward(test_img_noise)
        
        img_list.append(test_img)
        
        if epoch % param['display_interval'] == 0 or epoch == 0:
            print('Epoch[{}/{}] ({:.2f}%) '.format(epoch+1, num_epochs,\
                                                 ((epoch+1)/num_epochs)*100)+\
                  'Generator loss: {:.5} '.format(lossG) + \
                      'Discriminator loss: {:.5} '.format(lossD))
            winsound.Beep(1000,100) # Audio que to alert of the update.
        
    print('Final Loss, Generator: {:.7f}'.format(lossG) + \
          'Discriminator: {:.7f}'.format(lossD))
    
    return loss_vals


if __name__ == "__main__":
    
    start_time = time.time() # Get the start time.
    
    ## Determine if there is a GPU to use for training.
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    
    device = torch.device(dev)
    
    ## Setting the default data type to double precision to increase accuracy.
    ## Create all tensors on the default device.
    torch.set_default_device(device)
    torch.set_default_dtype(torch.float64)
    
    ## Determine the root path for the files.
    path = os.path.dirname(__file__) + '\\'
    
    ## Create arguments that are needed for the training. This arguments can be
    ##  changed from the defaults in the command line.
    parser = argparse.ArgumentParser(description='Training of a GAN')
    parser.add_argument('--param', default=path+'param.json',type=str,
                        help='File path for the Json file with the hyperparameters')
    parser.add_argument('--data-path', default=path+'Dataset\\64imagesReduced0.pickle',
                        type=str, help='Location of the training images')
    parser.add_argument('--model-name', default=path+'64BirdGAN.pkl',
                        type=str, help='Name to save the model as.')
    parser.add_argument('--batch', default='True',type=bool,
                        help='Boolean value to use batching.')
    parser.add_argument('--shuffle', default='True',type=bool,
                        help='Boolean value to shuffle the data.')
    args = parser.parse_args()
    
    ## Open the hyperparameter file.
    with open(args.param) as paramfile:
        param = json.load(paramfile)
    paramfile.close()
    
    ## Generate the models and data.
    data, gen, disc = prep(param, device)
    print("Data Tenosrs and models constructed")
    print("Beginning Training.....")
    
    ## Run the model through training on the data.
    loss_vals = run(param['exec'], gen, disc, data, device)
    
    
    