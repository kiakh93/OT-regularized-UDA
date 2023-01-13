

import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader


class ImageDataset(Dataset):
    def __init__(self, root,domain, lr_transforms=None, hr_transforms=None):
 
        dataset = get_dataset(dataset="camelyon17",root_dir=root, download=False)
        # Get the training set
        self.train_data = dataset.get_subset(
            domain
        )
        self.domain = domain

       

    def __getitem__(self, index):
        
        img,y,meta = self.train_data[index%len(self.train_data)]
        img = np.array(img,dtype = 'float32')[:,:,:]/255
        
        Transforms = [  
                        transforms.ToTensor(),
                        transforms.Normalize((.5,0.5,0.5), (0.5,0.5,0.5))
                        ]
    
        T = transforms.Compose(Transforms)
        img = T(img)

        return img,y,self.domain

    def __len__(self):
        return len(self.train_data)
