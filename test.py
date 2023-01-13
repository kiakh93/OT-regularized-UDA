import argparse
import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from datasets import *
import torch.nn as nn
import torch
from torchvision import models


def Eval(epoch,exp,domain,root): 

    
    os.makedirs('Eval_res'+str(exp)+'/' , exist_ok=True)

    # models
    net = models.resnet50(num_classes = 2,norm_layer=nn.InstanceNorm2d)
    net.load_state_dict(torch.load(os.path.join('saved_models_'+str(exp), str(epoch)+'.pth')))
     
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() >= 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = net.to(device)  
    test_loaderA = DataLoader(ImageDataset(root = root, domain = domain, lr_transforms=None, hr_transforms=None),
                        batch_size=32, shuffle=True, num_workers=16)
    
    test_loader_iteratorA = iter(test_loaderA)
    

    net.eval()
    count = 0

    with torch.no_grad():
        for i in range(len(test_loaderA)):
            t =  i
            try:
                X1,Y1,D1= next(test_loader_iteratorA)
            except StopIteration:
                train_loader_iteratorA = iter(test_loaderA)
                X1,Y1,D1 = next(test_loader_iteratorA)

            X1 = X1.to(device)             
            Y1 = Y1.to(device)             
            output = net(X1)     
            pred_y = output.cpu().detach().numpy()
            pred_y = np.argmax(pred_y, axis=1)
            for i in range(len(pred_y)):
                if pred_y[i] == Y1[i].data.cpu().numpy():
                    count += 1            
    
        np.save('Eval_res'+str(exp)+'/' +str(epoch)+"_"+ domain +'.npy',count)




