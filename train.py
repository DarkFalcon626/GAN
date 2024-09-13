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
    
    gen.apply(src.weights_init)
    disc.apply(src.weights_init)
    
    return data, gen, disc


def run(param, gen, disc, data, device):
    
    ## Using the bineary cross entropy loss function to determine the loss values
    loss = nn.BCELoss()
    
    ## Using the ADAM optimizer to optimize both networks.
    optimizerD = optim.Adam(disc.parameters(), lr=param['learn_rate'], 
                            betas=(param['beta1'],param['beta2']))
    optimizerG = optim.Adam(gen.parameters(), lr=param['learn_rate'],
                            betas=(param['beta1'],param['beta2']))
    
    ## Create list to store loss at each epoch.
    loss_vals = []
    
    ## Get the number of epochs and batchs from the json file.
    n_epochs = param['num_epochs']
    n_batchs = param['num_batches']
    
    ## Run through the epochs.
    for epoch in range(n_epochs):
        loss_g = 0  # Initialize the losses for the batchs.
        loss_d = 0 
        
        ## Create the randomized batchs
        batchs = data.create_batches(n_batchs, device)
        
        ## Create the latent vector for generating images.
        noise = torch.randn(data.batch_size, gen.n_lat, 1, 1).to(device)
        
        for i, batch in enumerate(batchs):
            
            ## Generate the fake images.
            data_fake = gen.forward(noise)
            data_real = batch
            
            ## Train the discriminator.
            loss_d += disc.train(data_real, data_fake, loss, optimizerD, device)
            
            ## Generate the fake images again.
            data_fake = gen.forward(noise).detach()
            
            ## Train the generator.
            loss_g += gen.train(disc, data_fake, loss, optimizerG, device)

        ## Average out the losses from all the batchs.
        disc_loss = loss_d/n_batchs
        gen_loss = loss_g/n_batchs
        
        loss_vals.append([disc_loss, gen_loss])
        
        if epoch == 0 or (epoch+1)%param['display_epochs'] == 0:
            print('Epoch [{}/{}] ({:.1f})'.format(epoch+1, n_epochs, ((epoch+1)/n_epochs)*100) +\
                  '\tGen Loss: {:.5f}'.format(gen_loss) +\
                  '\tDisc Loss: {:.5}'.format(disc_loss))
            
            torch.save_image(data_fake, args.image_save, normalize=True)
    
    print('Final Gen loss: {:.7f}'.format(gen_loss))
    print('Fianl Disc Loss: {:.7f}'.format(disc_loss))
    
    loss_vals = np.array(loss_vals)
    loss_vals = np.transpose(loss_vals, (1,0))
    
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


##-----------------------------------------------------------------------------
## Main code
##-----------------------------------------------------------------------------

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
    parser.add_argument('--fig-name', default=path+"trainingData.png",
                        type=str, help='Name of the training data image')
    parser.add_argument('--image-save', default=path+"Images")
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
    
    
    