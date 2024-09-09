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
import argparse
import torchvision.transforms as trans
import PIL


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

def save_img(image, i):
    '''
    Consumes an tensor image and an integer i and saves the tensor as a PNG
    image.

    Parameters
    ----------
    image : Torch Tensor
        A torch tensor to be converted to an image.
    i : Int
        The number of the image in the batch.

    Returns
    -------
    None.
    '''
    
    transform = trans.ToPILImage()
    
    img = transform(image)
    img.save(args.save_path + f'gen_image{i}.png', 'PNG')
    
    
##-----------------------------------------
## Main Code
##-----------------------------------------

if __name__ == "__main__":
    
    ## Determine if there is a GPU to use.
    if  torch.cuda.is_available():
        dev = 'cuda:0'
    else:
        dev = 'cpu'
    
    device = torch.device(dev)
    
    ## Set the default data type and device.
    torch.set_default_device(device)
    torch.set_default_dtype(torch.float32)
    
    ## Determine the root path for the files.
    path = os.path.dirname(__file__) + '\\'
    
    ## Create arguments that are need for the generating of images.
    parser = argparse.ArgumentParser(description="Generating images")
    parser.add_argument('--gen-model', default=path+'64BirdGAN_generator.pkl',
                        help='The file path for the pickle file containing the generator model.',
                        type=str)
    parser.add_argument('--num-images', default = 1,
                        type=int, help='Number of images to generate.')
    parser.add_argument('--save', default='True',
                        help="Boolean to tell if the images should be saved.",
                        type=bool)
    parser.add_argument('--save-path', default = path + 'Images\\')
    args = parser.parse_args()

    ## Load in the model
    with open(args.gen_model, 'rb') as modelfile:
        model = pickle.load(modelfile)
    modelfile.close()
    
    ## Generate the latent vectors.
    z = torch.randn(args.num_images, model.n_lat, 1, 1)
    
    ## Generate the images.
    images = model.forward(z)
    
    ## Go through the batch and show and save each image.
    for i, image in enumerate(images):
        View(image)
        if args.save:
            save_img(image, i)
        
    
    
    