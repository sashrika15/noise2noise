#print('__file__={0:<35} | __name__={1:<20} | __package__={2:<20}'.format(__file__,__name__,str(__package__)))
from dataset import NoisyDataset
from noise2noise import Noise2Noise
from torch.utils.data import DataLoader

data_dir = '/Users/sashrikasurya/Documents/noise2noise-torch/data/test'
ckpt = '/Users/sashrikasurya/Documents/noise2noise-torch/weights/n2n-gaussian.pt'
root_dir = '/Users/sashrikasurya/Documents/noise2noise-torch/'
noise = 'gaussian'

dataset = NoisyDataset(data_dir, noise)
test_loader = DataLoader(dataset, batch_size=1)
n2n = Noise2Noise(root_dir)
n2n.load_model(ckpt)
n2n.test(test_loader)