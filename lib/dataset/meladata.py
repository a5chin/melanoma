import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from pathlib import Path
from PIL import Image

from lib.config import config


class MelaDataset(Dataset):
    def __init__(self, train=True):
        super().__init__()
        self.train = train
        self.root = config.DATASET.IMAGE_TRAIN
        self.transform = {
            'train': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomCrop(128),
                transforms.RandomAffine(degrees=(-30, 30), scale=(0.8, 1.2), resample=Image.BICUBIC),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }

        df = pd.read_csv(config.MODEL.IDS_TRAIN).dropna()
        benign = df[df['target'] == 0]
        malignant = df[df['target'] == 1]
        # malignant_upsampled = resample(
        #     malignant,
        #     replace=True,
        #     n_samples=len(benign),
        #     random_state=0
        # )
        benign_downsampled = resample(
            benign,
            replace=False,
            n_samples=len(malignant),
            random_state=0
        )
        data = pd.concat([benign_downsampled, malignant])
        traindata, valdata = train_test_split(data)
        traindata, valdata = traindata.reset_index(drop=True), valdata.reset_index(drop=True)

        self.df = traindata if self.train else valdata
        self.sex = {'male': 0, 'female': 1}
        self.age = {np.sort(pd.Series.unique(self.df['age_approx']))[i]: i for i in range(len(pd.Series.unique(self.df['age_approx'])))}
        self.site = {'oral/genital': 0, 'palms/soles': 1, 'head/neck': 2, 'lower extremity': 3, 'upper extremity': 4, 'torso': 5}
        self.labels = np.array(self.df['target'])

    def __getitem__(self, index):
        img = self.transform['train' if self.train else 'val'](
            Image.open(
                self.root / Path(self.df['image_name'][index] + '.jpg')
            )
        )
        label = torch.from_numpy(np.array(self.df['target'][index]))
        sex = torch.from_numpy(np.array(self.sex[self.df['sex'][index]]))
        age = torch.from_numpy(np.array(self.age[self.df['age_approx'][index]])) / config.DATASET.NUM_CLASSES_AGE
        site = torch.from_numpy(np.array(self.site[self.df['anatom_site_general_challenge'][index]]))
        site = site.to(torch.long)
        site = F.one_hot(site, num_classes=len(self.site))

        return img, sex, age, site, label

    def __len__(self):
        return len(self.labels)
