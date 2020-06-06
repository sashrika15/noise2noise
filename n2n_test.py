from dataset import NoisyDataset
from noise2noise import Noise2Noise
from torch.utils.data import DataLoader

#data_path = <Path to directory containing images>
data_path = "/Users/sashrikasurya/Documents/noise2noise-torch/data/test"

#noise types: gaussian, text
n2n = Noise2Noise(data_path,noise='gaussian',show=True)
