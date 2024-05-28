import os
import os.path as op
import time

import torch

from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import numpy as np
import ujson as json

from tqdm import tqdm

from torch.utils.data import Dataset
from torchvision.utils import save_image
from torchvision.io import read_image
from torchvision.io import ImageReadMode
import torchvision
import random

from PIL import Image



def prepare_alph_old(alph_pt: str) -> list:
    alphabet = []
    for a in json.load(open(alph_pt, "r"))["alphabet"]:
        alphabet.append(a[0])
    return alphabet


def prepare_alph(alph_pt: str) -> (list, dict):
    with open(alph_pt, "r") as alph_f:
        alph = json.load(alph_f)["alphabet"]
    alph = [i[0] for i in alph[0]]
    # alph.append("NONE")
    alph_dict = {alph[i]: i for i in range(len(alph))}
    # print(alph_dict)
    return alph, alph_dict


class DeepFakeDataset(Dataset):
    def __init__(self, data_dir, transforms):
        self.data_dir = data_dir

        self.all_files, self.all_classes = [], []
        self.files_per_classes = []
        self.data = []

        print("======= LOADING DATA(DeepFakeNetDataset) =======")
        start_time = time.time()

        # trans1 = torchvision.transforms.ToTensor()
        # trans2 = torchvision.transforms.Resize((37, 37), antialias=False)

        for root, dirs, files in sorted(os.walk(self.data_dir, topdown=False)):
            for f in files:
                self.all_files.append(os.path.join(root, f))

        print('DeepFakeNetDataset_length: ', len(self.all_files),
              '\nTime: {:.2f} sec'.format(time.time() - start_time))

    def __len__(self) -> int:
        return len(self.all_files)

    def __getitem__(self, idx: int) -> dict:
        # print(self.all_files[idx])
        image = Image.open(self.all_files[idx]).convert('RGB')
        trans1 = torchvision.transforms.ToTensor()
        transform = torchvision.transforms.Resize((256, 256))

        label = 0
        if '/real/' in self.all_files[idx]:
            label = 1

        sample = {
            "image": transform(trans1(image)),
            "label": torch.tensor(label)
        }

        return sample



