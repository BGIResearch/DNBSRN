import os
import glob
import random
import tifffile
import numpy as np
from torch.utils.data import Dataset, DataLoader


def train_valid_dataloader(opt):
    train_dataset = LH_dataset(opt, 't')
    t = DataLoader(train_dataset, batch_size=opt.batchsizeT, pin_memory=True, num_workers=0, shuffle=True)
    valid_dataset = LH_dataset(opt, 'v')
    v = DataLoader(valid_dataset, batch_size=opt.batchsizeV, pin_memory=True, num_workers=0, shuffle=False)
    return t, v


class LH_dataset(Dataset):
    def __init__(self, opt, dataset):
        self.imgpath = opt.imgLHdir
        self.dx = opt.imgTsize
        self.channel = opt.channel
        self.cyclist = os.listdir(opt.imgLHdir)
        self.cyclist.sort()
        if dataset == 't':
            self.ids = self.cyclist[:40]
            self.ndata = opt.nT
        elif dataset == 'v':
            self.ids = self.cyclist[40:]
            self.ndata = opt.nV

    def __getitem__(self, i):
        imgfile = glob.glob(fr'{self.imgpath}/{self.ids[i//len(self.channel)%len(self.ids)]}/*.{self.channel[i%len(self.channel)]}.*.tif')
        img = tifffile.imread(imgfile)
        d1, d2 = np.random.randint(img.shape[1]-self.dx+1), np.random.randint(img.shape[2]-self.dx+1)
        img = img[:, d1:d1+self.dx, d2:d2+self.dx]
        img = np.float32(img)
        range = random.uniform(0.9, 1.1)
        img[0], img[1] = img[0]/(range*np.percentile(img[0], 99)), img[1]/(range*np.percentile(img[1], 99))
        lr, hr = np.expand_dims(img[0], axis=0), np.expand_dims(img[1], axis=0)
        return lr, hr

    def __len__(self):
        return self.ndata
