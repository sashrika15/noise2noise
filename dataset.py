import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import random
import torchvision.transforms.functional as tvf
from PIL import Image, ImageDraw, ImageFont
from string import ascii_letters
import random

class NoisyDataset(Dataset):
    '''
    Loads dataset.
    NoisyDataset inherits from an abstract class representing Dataset
    '''

    def __init__(self, root_dir, noise):
        '''
        Initialise dataset
        '''
        self.root_dir = root_dir
        self.imgs = []
        self.imgs = [os.path.join(root_dir, i) for i in os.listdir(root_dir)]
        self.crop_size = 256
        self.noise = noise
    
    def gaussian_noise(self,img):
        '''
        Add Gaussian noise in dataset
        Input: img of type PIL.Image
        Output: Noisy image of type PIL.Image
        '''
        print("Image type: ")
        print(type(img))
        w,h = img.size
        c = len(img.getbands())

        sigma = np.random.uniform(20,50)
        gauss = np.random.normal(10,sigma,(h,w,c))
        noisy = np.array(img) + gauss
        
        #Values less than 0 become 0 and more than 255 become 255
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)

        return Image.fromarray(noisy)

    '''
    def text_overlay(self,img):
        w,h = img.size
        c = len(img.getbands())
        
        
        font_type = ImageFont.truetype('Arial.ttf',np.random.randint(10,40))
        len_text = np.random.randint(4,20)
        text = ''.join(random.choice(ascii_letters) for i in range(len_text))
        #print(text)
        x = np.random.randint(0,w)
        y = np.random.randint(0,h)
        col = tuple(np.random.randint(0,255,c))
        im = img.copy()
        draw = ImageDraw.Draw(im)
        for i in range(random.randint(0,20)):
            draw.text((x,y),text,col,font=font_type)

        return im
    '''

    def _add_text_overlay(self, img):
        """Adds text overlay to images."""

        self.noise_param = 0.3
        w, h = img.size
        c = len(img.getbands())

        # Choose font and get ready to draw
        text_img = img.copy()
        text_draw = ImageDraw.Draw(text_img)

        # Text binary mask to compute occupancy efficiently
        w, h = img.size
        mask_img = Image.new('1', (w, h))
        mask_draw = ImageDraw.Draw(mask_img)

        # Random occupancy in range [0, p]
       
        max_occupancy = np.random.uniform(0, self.noise_param)
        
        def get_occupancy(x):
            y = np.array(x, dtype=np.uint8)
            return np.sum(y) / y.size

        # Add text overlay by choosing random text, length, color and position
        while 1:
            font = ImageFont.truetype('Arial.ttf', np.random.randint(16, 21))
            length = np.random.randint(10, 25)
            chars = ''.join(random.choice(ascii_letters) for i in range(length))
            color = tuple(np.random.randint(0, 255, c))
            pos = (np.random.randint(0, w), np.random.randint(0, h))
            text_draw.text(pos, chars, color, font=font)

            # Update mask and check occupancy
            mask_draw.text(pos, chars, 1, font=font)
            if get_occupancy(mask_img) > max_occupancy:
                break

        return text_img


    
    def square_image(self,img):
        '''
        Randomly crops the image to a square of size (256,256)
        Input: img of type PIL.Image
        Output: Cropped img of type PIL.Image
        '''
        
        w,h = img.size
        if min(w, h) < self.crop_size:
            img = tvf.resize(img, (self.crop_size+1, self.crop_size+1))
            w,h = img.size
    
        i = np.random.randint(0, h - self.crop_size + 1)
        j = np.random.randint(0, w - self.crop_size + 1)

        img = tvf.crop(img, i, j, self.crop_size, self.crop_size)

        return img
    
    def __len__(self):
        '''
        Returns length of dataset
        '''
        return len(self.imgs)

    def __getitem__(self,index):
        '''
        Compiles dataset
        '''
       
        img_path = self.imgs[index] 
        img =  Image.open(img_path).convert('RGB')
        resized_img = self.square_image(img)
        if self.noise == 'text':
            source = tvf.to_tensor(self._add_text_overlay(resized_img))
        else:
            source = tvf.to_tensor(self.gaussian_noise(resized_img))
        
        #print("resized = ", str(resized_img.size))
        target = tvf.to_tensor(resized_img).unsqueeze(0)
        #print("target = ", str(target.size()))
        #resized_img = self._random_crop([img])[0]
        #source = tvf.to_tensor(self.gaussian_noise(resized_img))
        #source = tvf.to_tensor(self._add_text_overlay(resized_img))
        

        return source, target


 