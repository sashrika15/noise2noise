import os
import torch
import torchvision.transforms.functional as tvf
from unet import Unet

class Noise2Noise:
    def __init__(self,root_dir):
        '''
        Initialise class
        '''
        print("Initialising Noise2Noise Model")
        self.root_dir = root_dir
        self.model = Unet(in_channels=3)
    
    def load_model(self,ckpt):   
        print("Loading model from saved Checkpoint")
        self.model.load_state_dict(torch.load(ckpt, map_location='cpu'))


    def test(self,test_loader):
        '''
        Inference of model
        Input: test_loader: Dataloader object
        '''

        source_imgs = []
        denoised_imgs = []

        #Directory for output of model
        denoised_dir = os.path.dirname(self.root_dir)
        save_path = os.path.join(denoised_dir, 'Output')
        if not os.path.isdir(save_path):
            print("Making dir for denoised images")
            os.mkdir(save_path)

        for _, (source, _) in enumerate(test_loader):

            source_imgs.append(tvf.to_pil_image(torch.squeeze(source))) 
            output = self.model(source)
            denoised = tvf.to_pil_image(torch.squeeze(output))
            denoised_imgs.append(denoised)
    
        #Save images to directory
        for i in range(len(source_imgs)):
            source = source_imgs[i]
            denoised = denoised_imgs[i]
            source.save(os.path.join(save_path,'source{}.png'.format(i)))
            denoised.save(os.path.join(save_path,'denoised{}.png'.format(i)))
            

