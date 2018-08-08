from os import listdir
from os.path import join

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
from torch.utils import data
from torchvision.transforms import Compose, RandomHorizontalFlip, Resize,  CenterCrop, Normalize, ToTensor


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def train_transform(crop_size, image_size):
    return Compose([
        RandomHorizontalFlip(),
        CenterCrop(crop_size),
        Resize(image_size),
        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])


class MyDataset(Dataset):
    def __init__(self, dataset_dir, crop_size, image_size):

        self.lr_path = dataset_dir + '/profile/'
        self.hr_path = dataset_dir + '/frontal/'
        self.transform = train_transform(crop_size, image_size)
        self.lr_filenames = [join(self.lr_path, x) for x in listdir(self.lr_path) if is_image_file(x)]
        self.hr_filenames = [join(self.hr_path, x) for x in listdir(self.hr_path) if is_image_file(x)]

    def __getitem__(self, index):

        batch = {}
        batch['img_profile'] = self.transform(Image.open(self.lr_filenames[index]))
        img_frontal_name = self.lr_filenames[index].split('/')
        img_frontal = img_frontal_name[-1].split('_')
        batch['label'] = int(img_frontal[0])
        batch['img_frontal'] = self.transform((Image.open(self.hr_filenames[index])))

        return batch

    def __len__(self):
        return len(self.lr_filenames)
