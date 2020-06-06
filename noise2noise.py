import os
import torch
import torchvision.transforms.functional as tvf
from torch.utils.data import DataLoader

from unet import Unet
from dataset import NoisyDataset

root_dir = os.path.dirname(os.path.realpath(__file__))

class Noise2Noise:
    '''
    Noise2Noise class. 
    '''
    def __init__(self,data_dir,noise,show=False):
        '''
        Initialise class
        '''
        print("Initialising Noise2Noise Model")

        self.noise = noise
        self.data_dir = data_dir
        self.show = show
        test_loader = self.load_dataset()
        if torch.cuda.is_available():
            self.map_location = 'cuda'
        else:
            self.map_location = 'cpu'

        try:
            self.model = Unet(in_channels=3)
            self.load_model()
        except Exception as err:
            print("Error at {}".format(err))
            exit()

        self.inference(test_loader)
    
    def load_model(self):   
        if self.noise == "text":
            ckpt_dir = root_dir + "/weights/n2n-text.pt"
        else:
            ckpt_dir = root_dir + "/weights/n2n-gaussian.pt"

        self.model.load_state_dict(torch.load(ckpt_dir, self.map_location))
        print("Loaded model from saved checkpoint")

    def load_dataset(self):
        print("Creating Dataset")
        dataset = NoisyDataset(self.data_dir, self.noise, crop_size=256)
        test_loader = DataLoader(dataset, batch_size=1)
        return test_loader

    def inference(self,test_loader):
        '''
        Inference of model
        Input: test_loader: Dataloader object
        '''
        print("Running inference")
        source_imgs = []
        denoised_imgs = []

        #Directory for output of model
        save_path = os.path.join(root_dir, 'Output')
        if not os.path.isdir(save_path):
            print("Making dir for denoised images")
            os.mkdir(save_path)

        for source in list(test_loader):
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
            if self.show==True:
                source.show()
                denoised.show()
            
            

