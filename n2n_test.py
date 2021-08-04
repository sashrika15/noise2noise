from model import Noise2Noise

#data_path = <Path to directory containing images>

data_path = "./test_images"

#noise types: gaussian, text
n2n = Noise2Noise(data_path,noise='text')
