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
import numpy as np
import pylab as plt


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
    '''
    Consumes a generator and discriminator model along with a data object and
    torch device to train the models on.

    Parameters
    ----------
    param : JSON
        Parameters for the training.
    gen : Net object
        The generator network to generate images.
    disc : Net object
        The discriminator for deciding if an image is real or fake.
    data : Data object
        The processed data for training of the GAN.
    device : Torch device
        Device to store the models and data on.

    Returns
    -------
    loss_vals : Array
        The numpy array containing the values for loss in each epoch.

    '''
    
    ## Using the bineary cross entropy loss function to determine the loss values
    loss = nn.BCELoss()
    
    ## Using the ADAM optimizer to optimize both networks.
    optimizerD = optim.Adam(disc.parameters(), lr=param['learn_rate'], 
                            betas=(param['beta1'],param['beta2']))
    optimizerG = optim.Adam(gen.parameters(), lr=param['learn_rate'],
                            betas=(param['beta1'],param['beta2']))
    
    loss_vals = []
#    img_list = []
    
    num_epochs = int(param['num_epochs'])
    for epoch in range(num_epochs):
        
        n_batches = param['num_batches']
        batches, real_labels = data.create_batches(n_batches, device, args.shuffle)
        batches = batches.float()
        real_labels.float()

        fake_labels = torch.zeros(data.batch_size, dtype=torch.float32).to(device)
        
        batch_errorD = 0
        batch_errorG = 0
        
        for i, batch in enumerate(batches, 0):
            
            ## Train the Generator
            ## -------------------
            
            optimizerD.zero_grad()
            
            ## Create the latent vector for the generator input.
            z = torch.randn(data.batch_size, gen.n_lat, 1, 1, dtype=torch.float,
                            device=device)
            
            ## Generate the fake images from the latent vector.
            gen_imgs = gen.forward(z)
            
            ## Determine the discriminators guesses at the fake images.
            output = disc.forward(gen_imgs).view(-1)
            
            ## Determine how good those guess's where.
            g_loss = loss(output, real_labels)
            
            ## update the generator model.
            g_loss.backward()
            optimizerG.step()
            
            ## Train the Discriminator
            ## -----------------------
            
            optimizerD.zero_grad()
            
            ## Determine the discrimintors guesses at the real images.
            output = disc.forward(batch).view(-1)
            
            ## Determine how good those guess's are.
            real_loss = loss(output, real_labels)
            
            ## Determine the guess's at the fake images.
            output = disc.forward(gen_imgs.detach()).view(-1)
            
            ## Determine how good those guess's are.
            fake_loss = loss(output, fake_labels)
            
            ## Taking the average of the real and fake loss values.
            d_loss = (real_loss + fake_loss)/2 
            
            ## Update the discriminator model.
            d_loss.backward()
            optimizerD.step()
            
            ## Add the error to the total batch errors.
            batch_errorD += d_loss.item()
            batch_errorG += g_loss.item()
            
            ## Delete the graphs and data from memory to save memory space.
            del z
            del gen_imgs
            del output
            del g_loss, d_loss
            del real_loss, fake_loss
            
        ## Determine the average error of the batches.
        lossD = batch_errorD/n_batches
        lossG = batch_errorG/n_batches
        
        loss_vals.append([lossG,lossD])
        
        if (epoch+1) % param['display_interval'] == 0 or epoch == 0:
            print('Epoch[{}/{}] ({:.2f}%) '.format(epoch+1, num_epochs,\
                                                 ((epoch+1)/num_epochs)*100)+\
                  'Generator loss: {:.5} '.format(lossG) + \
                      'Discriminator loss: {:.5} '.format(lossD))
            winsound.Beep(1000,100) # Audio que to alert of the update.
        
        del lossD, lossG
        
    print('Final Loss, Generator: {:.7f}'.format(loss_vals[0][-1]) + \
          'Discriminator: {:.7f}'.format(loss_vals[1][-1]))
    
    ## Convert the loss values to an numpy array and transpose the array.
    loss_vals = np.array(loss_vals)
    loss_vals = np.transpose(loss_vals,(1,0))
    
    return loss_vals


def save_value(save):
    '''
    Determines if the input is a excepted value to a yes or no question, if not
    get a new input.

    Parameters
    ----------
    save : STR
        Input string to a yes or no question.

    Returns
    -------
    save : Bool
        Awnser to the yes or no question.
    '''
    
    ## Reduce any uppercase to lowercase to maintain the meaning of the word.
    save = save.lower()
    
    ## Determine if the awnser is true or false.
    if save in ['true', '1', 'yes']:
        save = True
    elif save in ['false', '0', 'no']:
        save = False
    else: # If the awnser is not an excepted value ask the question again.
        save = input('Value entered is not a proper response. Please enter \
                     either true, 1, yes or false, 0, no. ->')
        save = save_value(save) # Check the new awnser.
    
    return save


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
    torch.set_default_dtype(torch.float32)
    
    ## Determine the root path for the files.
    path = os.path.dirname(__file__) + '\\'
    
    ## Create arguments that are needed for the training. This arguments can be
    ##  changed from the defaults in the command line.
    parser = argparse.ArgumentParser(description='Training of a GAN')
    parser.add_argument('--param', default=path+'param.json',type=str,
                        help='File path for the Json file with the hyperparameters')
    parser.add_argument('--data-path', default=path+'Dataset\\64images.pickle',
                        type=str, help='Location of the training images')
    parser.add_argument('--model-name', default=path+'64BirdGAN.pkl',
                        type=str, help='Name to save the model as.')
    parser.add_argument('--batch', default='True',type=bool,
                        help='Boolean value to use batching.')
    parser.add_argument('--shuffle', default='True',type=bool,
                        help='Boolean value to shuffle the data.')
    parser.add_argument('--fig_name', default=path+"trainingData.png",
                        type=str, help='Name of the training data image')
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
    
    train_time = time.time() - start_time # Determine the time took to train.
    
    if (train_time//3600) > 0:
        hours = train_time//3600
        mins = (train_time - hours*3600)//60 
        secs = train_time - hours*3600 - mins*60
        
        print('The model trained in {} hours, {} mins and {:.0f} second'.format(hours, mins, secs))
    elif (train_time//60) > 0:
        print('The model trained in {} mins and {:.0f} seconds'.format(train_time//60,
                                                                       train_time-(train_time//60)*60))
    else:
        print('The model trained in {:.0f} seconds'.format(train_time))
    
    x = np.arange(1,len(loss_vals[0])+1) # Axis for the number of epochs.
    
    plt.plot(x, loss_vals[0], label="Generator Loss")
    plt.plot(x, loss_vals[1], label="Discriminator Loss")
    plt.title("Loss per Epoch.")
    plt.grid()
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    
    ## Asks the user if the model should be saved or not.
    save = input('Do you want to save the model and training data. -> ')
    save = save_value(save)
    
    ## If save is true save the model as a pickle file.
    if save:
        generator_name = args.model[:-4] + "_generator" + args.model[-4:]
        discriminator_name = args.model[:-4] + "_discriminator" + args.model[-4:]
        
        with open(generator_name, 'wb') as f:
            pickle.dump(gen, f) # Save the generator.
        f.close()
        
        with open(discriminator_name, 'wb') as f:
            pickle.dump(disc, f)
        f.close()
        
        plt.savefig(args.fig_name, format='png')
        
        print("The generator and discriminator are saved as {} and {}".format(
            args.generator_name, discriminator_name))
    
    
    