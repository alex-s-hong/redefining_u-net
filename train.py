import matplotlib.pyplot as plt
import torch
import torchvision
import lesion
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
import utils as util
import torch.nn.functional as F
import residual_unet


print("Running~")

class ToTensor_segmap(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, segmap):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        # In this case, Channels is 1, so there is no need to swap since data is in HxW      

        segmap = np.array(segmap)
        return torch.from_numpy(segmap) / 255

    
#mean and std of dataset when grayscaled [0.6193, 0.6193, 0.6193], std = [0.1547, 0.1547, 0.1547]
#image net mean and std  mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

image_transform = transforms.Compose([
        transforms.RandomGrayscale(1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]) ])
seg_transform = ToTensor_segmap()


train_dataset = lesion.LesionDataset("data",joint_transform=True,img_transform=image_transform, seg_transform=seg_transform)
val_dataset = lesion.LesionDataset("data",img_transform=image_transform, seg_transform=seg_transform,folder_name = 'val')
train_loader = DataLoader(train_dataset, batch_size=8,
                        shuffle=True, num_workers=8)
val_loader =  DataLoader(val_dataset, batch_size=8,
                        shuffle=True, num_workers=8)


model = residual_unet.film_unet(dropout = "all")

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)

model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4,weight_decay = 1e-05) 
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)


x = np.array([])
y = np.array([])

best_loss = 100
model_name = "film_all.pth"
for epoch in range(30):
    running_loss = 0.0

    model.train()
    
    for i,data in enumerate(train_loader):
        image = data[0].to(device)
        segmap = data[1].to(device).long()
        output = model(image)
        
        optimizer.zero_grad()
        loss = criterion(output, segmap)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() / len(train_loader)
        
    avg_val_loss = 0
    
    model.eval()

    for i,data in enumerate(val_loader):
        image = data[0].to(device)
        segmap = data[1].to(device).long()
        output = model(image)
        loss = criterion(output, segmap)
        avg_val_loss += loss.item() / len(val_loader) # Loss.item() is already averaged per batch
    if(avg_val_loss < best_loss):
        best_loss = avg_val_loss
        torch.save(model,"models/" + model_name)
    
    scheduler.step()
            
        
    x = np.append(x,running_loss)
    y = np.append(y,avg_val_loss)
    print("Epoch:{} Loss:{} Val_loss:{}  Best_val_loss:{}".format(epoch,running_loss,avg_val_loss,best_loss))
print("Finished Training!")
np.savez_compressed("result/film_all",x=x,y=y)
