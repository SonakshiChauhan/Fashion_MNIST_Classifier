# Titanic_Survival

**Completed by Sonakshi Chauhan.**

**Overview:** In this project we are using the pre-existing pytorch dataset named 'FashionMNIST'

-Building a classifier to classify the dataset having 28x28 grayscale images into 10 different categories
-Building neural network using PyTorch
-Visualization for a better understanding of data
-Training, and testing followed by accuracy analysis of the model 

**Data:** Fashion MNIST datasets , downloaded using '''bash
datasets.FashionMNIST''' 

**Deliverables:** Categorial Classification

**Let's Shop**

![image](https://user-images.githubusercontent.com/91408631/236686235-316bb122-58f5-4519-a08d-3288b09283f8.png)


## Topics Covered
1. PyTorch framework
2. Model Building and understanding various layers
3. More about custom loss functions
4. Classification understanding and implementation

## Tools Used
1. Scikit-learn
2. Google Colab
3. TorchVision
4. PyTorch

## Installation and Usage

#### Jupyter Notebook - to run ipython notebook (.ipynb) project file
Follow the instructions on https://docs.anaconda.com/anaconda/install/ to install Anaconda with Jupyter. 
Alternatively:
VS Code can render Jupyter Notebooks
## Notebook Structure
The structure of this notebook is as follows:
 -Imports
 -Data Loading and Vizualization
 -Data Pre-processing
 -Defining Model using Pytorch
 -Model Training
 -Model Saving and using validation set to find best model
 -Testing and Evaluation

# Data Pre-Processing
```bash
#standard pytorch modules
import torch #holds all the things required for tensor computation
import torch.nn as nn #provides classes and functions to build nn
import torch.nn.functional as F
import torch.optim as optim #for optimizers like ADAM etc
from torch.utils.tensorboard import SummaryWriter #generates report for tensor board
from torch.autograd import Variable
#import torchvision module to handle image manipulation
import torchvision #has datasets, model architectures # image transforma for CV
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

#calculate train time , writing train data to files etc.
import time
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from IPython.display import clear_output

torch.set_printoptions(linewidth=120)
torch.set_grad_enabled(True)
from sklearn.metrics import confusion_matrix
import torchvision.datasets as datasets```

-> Above are the libraries/packages that need to be imported before heading ahead

# Data loading and vizualization
```bashtest_set=datasets.FashionMNIST(
    train=False,
    root='./data',
    download=True,
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,),)])
)

train_set=datasets.FashionMNIST(
    train=True,
    root='./data',
    download=True,
    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,),)])
)

indices=list(range(len(train_set)))
np.random.shuffle(indices)
split=int(np.floor(0.2*len(train_set)))
train_sample=SubsetRandomSampler(indices[:split])
valid_sample=SubsetRandomSampler(indices[split:])

train_loader=torch.utils.data.DataLoader(train_set,sampler=train_sample,batch_size=64)
valid_loader=torch.utils.data.DataLoader(train_set,sampler=train_sample,batch_size=32)
testloader=torch.utils.data.DataLoader(test_set,batch_size=64,shuffle=True)
```
->We firstly load the dataset
->After loading the datset we split in into train and test dataset
->Then using dataloader we load the datasets.

```bash
%matplotlib inline

dataiter = iter(train_loader)
print(dataiter)
images,labels=next(dataiter)

fig=plt.figure(figsize=(15,5))
for idx in np.arange(20):
  ax=fig.add_subplot(4,(int(20/4)),idx+1,xticks=[],yticks=[])
  ax.imshow(np.squeeze(images[idx]),cmap='gray')
  ax.set_title(labels[idx].item())
  fig.tight_layout
  ```
->Now we visualize the dataset and the output is as below:


![image](https://user-images.githubusercontent.com/91408631/236688049-50cfd106-4b0a-4cb2-aa14-a94cdf8a3cdd.png)


# Building Neural Network
```bash
class Classifier(nn.Module):
  def __init__(self):
    super().__init__()
    self.fc1=nn.Linear(784,256)
    self.fc2=nn.Linear(256,100)
    self.fc3=nn.Linear(100,64)
    self.fc4=nn.Linear(64,10)
    #defining the 20% dropout
    self.dropout=nn.Dropout(0.2)

  def forward(self,x):
    x=x.view(x.size(0),-1)
    x=self.dropout(F.relu(self.fc1(x)))
    x=self.dropout(F.relu(self.fc2(x)))
    x=self.dropout(F.relu(self.fc3(x)))
    #not using dropout on output layer
    x=F.log_softmax(self.fc4(x),dim=1)
    return x
```

->Above is the classifier model built using PyTorch.

#Model Training
```bash
model=Classifier()
#defining the loss function
criterion=nn.NLLLoss()
optimizer=optim.SGD(model.parameters(),lr=0.01)

valid_loss_min=np.Inf
epochs=40
steps=0
model.train()
train_losses,valid_losses=[],[]
for e in range(epochs):
  running_loss=0
  valid_loss=0
  #train the model
  for images,labels in train_loader:

    optimizer.zero_grad()
    log_ps=model(images)
    loss=criterion(log_ps,labels)
    loss.backward()
    optimizer.step()
    running_loss+=loss.item()*images.size(0)

  for image,labels in valid_loader:
    log_ps=model(images)
    loss=criterion(log_ps,labels)
    valid_loss+=loss.item()*images.size(0)

  running_loss=running_loss/len(train_loader.sampler)
  valid_loss=valid_loss/len(valid_loader.sampler)
  train_losses.append(running_loss)
  valid_losses.append(valid_loss)

  print(e+1,running_loss,valid_loss)
  if valid_loss<=valid_loss_min:
    torch.save(model.state_dict(),'model.pt')
    valid_loss_min=valid_loss

```
->Here we start the training process
->We track the training and validation loss using custom built function.
->Below is the output of training stacking the training and validation loss

![image](https://user-images.githubusercontent.com/91408631/236688365-665a6572-c9b8-47ec-ae94-3b5c5bbcb435.png)

->Vizualizing the above loss results we geta plot like below:

![image](https://user-images.githubusercontent.com/91408631/236688411-4eefc848-bfff-4576-b9b5-71a5afa0ea7a.png)


#Testing and Rvaluation
```bash
#track the test loss
test_loss=0
class_correct=list(0. for i in range(10))
class_total=list(0. for i in range(10))

model.eval()
for images,labels in testloader:
  #forward pass
  output=model(images)
  #calculate the loss
  loss=criterion(output,labels)
  #update the test loss
  test_loss+=loss.item()*images.size(0)
  #convert output probabilities to predicted class
  _, pred=torch.max(output,1)
  #compare predictions to true labels
  correct=np.squeeze(pred.eq(labels.data.view_as(pred)))
  #calculate test accuracy for each object class
  for i in range(len(labels)):
    label=labels.data[i]
    class_correct[label]+=correct[i].item()
    class_total[label]+=1

#calculate and print test loss
test_loss=test_loss/len(test loader.sampler)
print('Test Loss: {:.6f}\n'.format(test_loss))

for i in range(10):
  if class_total[i]>0:
    print('Test Accuracy of %5s: %2d%% (%2d/%2d)'%(str(i),100*class_correct[i]/class_total[i],
                                                   np.sum(class_correct[i]),np.sum(class_total[i])))
  
print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)'%(100. *np.sum(class_correct)/np.sum(class_total),
                                                    np.sum(class_correct),np.sum(class_total)))
```
->Now we finally test our model and stack the test accuracy having an overall accuracy as displayed below:

![image](https://user-images.githubusercontent.com/91408631/236688621-abccc18d-23ce-4422-974b-eee3f4afd385.png)



#Final Steps
```bash
#obtain one batch of test images
dataiter=iter(test loader)
images,lables=next(dataiter)

#get sample outputs
output=model(images)
#convert output probabilities to predicted classes
_,preds=torch.max(output,1)
#prep images for display
images=images.numpy()

fig=plt.figure(figsize=(25,4))
for idx in np.arange(16):
  ax=fig.add_subplot(2,int(20/2),idx+1,xticks=[],yticks=[])
  ax.imshow(np.squeeze(images[idx]),cmap='gray')
  ax.set_title("{} ({})".format(str(preds[idx].item()),str(labels[idx].item())),
                color=("green" if preds[idx]==labels[idx] else "red"))
                ```
->Now we visualize the and test the model on a batch of 16 photos and the results obtained are displayed below:

![image](https://user-images.githubusercontent.com/91408631/236688736-b78ecf15-aa02-470f-b400-55fc1d753312.png)

#Conclusion
->We built a Classifier for the Fashion MNIST dataset

**Contact:** sonakshichauhan1402@gmail.com

## Project Continuity
This project is complete

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. 

