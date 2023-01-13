import argparse
import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets
from geomloss import SamplesLoss
from datasets import *
import torch.nn as nn
import torch
from test import *
from torchvision import models

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
    parser.add_argument('--n_epochs', type=int, default=5, help='number of epochs of training')
    parser.add_argument('--ref_dataset', type=str, default=0, help='refrence dataset')
    parser.add_argument('--batch_size', type=int, default=32, help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.0001, help='adam: learning rate')
    parser.add_argument('--Lambda', type=float, default=0.1, help='lambda value')
    parser.add_argument('--exp', type=int, default=0, help='experiment id')
    parser.add_argument('--b1', type=float, default=0.599, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.9, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--n_cpu', type=int, default=16, help='number of cpu threaDSA to use during batch generation')
    opt = parser.parse_args()
    print(opt)

    os.makedirs('saved_models_'+ str(opt.exp) +'/'  , exist_ok=True)
    
    # Loss functions
    criterion = nn.CrossEntropyLoss()   
    loss_geom = SamplesLoss("sinkhorn", p=2, blur=.01,scaling =.95,verbose=False)
    # models
    net = models.resnet50(num_classes = 2,norm_layer=nn.InstanceNorm2d)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() >= 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = net.to(device) 
        
    #Source domain (train set)
    train_loaderA = DataLoader(ImageDataset(root = "../camelyon_original/data/", domain = "train", lr_transforms=None, hr_transforms=None),
                        batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu,drop_last=True)
    train_loader_iteratorA = iter(train_loaderA)
    
    #Target domain (val set)
    train_loaderB = DataLoader(ImageDataset(root = "../camelyon_original/data/", domain = "val", lr_transforms=None, hr_transforms=None),
                        batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu,drop_last=True)
        
    train_loader_iteratorB = iter(train_loaderB)
    

    optimizer = torch.optim.SGD(net.parameters(), lr=0.003, momentum=0.9, weight_decay=0.001)
    torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    Loss_G = []
    Loss_CE = []
    A_t = []
    A_v = []
    Lambda = opt.Lambda
    for epoch in range(opt.epoch, opt.n_epochs):
        net.train()
        for i in range(len(train_loaderA)):
            net.train()
            t = epoch*len(train_loaderA) + i
            try:
                X1,Y1,D1= next(train_loader_iteratorA)
            except StopIteration:
                train_loader_iteratorA = iter(train_loaderA)
                X1,Y1,D1 = next(train_loader_iteratorA)

            try:
                X2,Y2,D2= next(train_loader_iteratorB)
            except StopIteration:
                train_loader_iteratorB = iter(train_loaderB)
                X2,Y2,D2 = next(train_loader_iteratorB)
            X1 = X1.to(device)             
            Y1 = Y1.to(device)             
            X2 = X2.to(device)             
            Y2 = Y2.to(device)                
            # zero the parameter gradients
            optimizer.zero_grad()

            # Source data prediction
            outputs1 = net(X1)
            
            # Source features
            feat1 = nn.Sequential(*list(net.children())[:-1])(X1)
            
            # Target features
            feat2 = nn.Sequential(*list(net.children())[:-1])(X2)
            
            # OT loss
            loss_g = loss_geom(feat1.detach().squeeze(),feat2.squeeze())  
            
            # CE loss
            loss_c = criterion(outputs1, Y1)
            
            Loss_G.append(loss_g.item())
            Loss_CE.append(loss_c.item())
            loss_t = loss_c + loss_g*Lambda
            loss_t.backward()
            optimizer.step()
            
            # Accuracy of source
            pred_y = outputs1.cpu().detach().numpy()
            pred_y = np.argmax(pred_y, axis=1)
            acc = 0
            for i in range(len(pred_y)):
                if pred_y[i] == Y1[i].data.cpu().numpy():
                    acc += 1    
            
            output = net(X2)  
            
            # Accuracy of target
            pred_y = output.cpu().detach().numpy()
            pred_y = np.argmax(pred_y, axis=1)
            acc_v = 0
            for i in range(len(pred_y)):
                if pred_y[i] == Y2[i].data.cpu().numpy():
                    acc_v += 1  
            A_t.append(acc)
            A_v.append(acc_v)
            print("bstch_done: ", t,"|", "\t loss_CE: ", loss_c.item(),
                   "\t lossgeom: ", loss_g.item(), "\t acc: ", acc/opt.batch_size, "\t acc_v: ", acc_v/opt.batch_size
                     )   
            
         

        torch.save(net.state_dict(),'saved_models_'+ str(opt.exp) +'/' +str(epoch)+'.pth')
        
        # Evaluation for each train, val, and test set
        Eval(epoch,opt.exp,"train",opt.dataset_dir)
        Eval(epoch,opt.exp,"val",opt.dataset_dir)
        Eval(epoch,opt.exp,"test",opt.dataset_dir)


if __name__ == '__main__':
    
    main()


