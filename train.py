from torch.utils.data import Dataset
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from dataset import get_train_test_loaders

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,6,kernel_size=3)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,6,kernel_size=3)
        self.conv3 = nn.Conv2d(6,16,kernel_size=3)
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,48)
        self.fc3 = nn.Linear(48,25)
        self.relu = nn.ReLU()
        

    def forward(self,x):
        x = self.relu(self.conv1(x))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(-1,16*5*5)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train(model,loss_func,optimizer,train_loader,epoch):
    running_loss = 0.0
    
    for i , data in enumerate(train_loader,0):
        inputs = Variable(data['image'].float())
        labels = Variable(data['label'].long())

        #forward propogation + backward + optimize
        outputs = model(inputs)
        loss = loss_func(outputs,labels[:,0])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        #print stats
        running_loss += loss.item()
        if i % 100 == 0:
            print(' [{} , {}] , loss : {:.4f} '.format(epoch,i,running_loss/(i+1)))
    loss_per_epoch = running_loss/len(train_loader)
    return loss_per_epoch


def plot_losses(loss_arr):
    plt.plot(loss_arr,'-x')
    plt.xlabel('No. of epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epochs')
    plt.show(block=False)


def main():
    model = ConvNet().float()
    loss_func = nn.CrossEntropyLoss()
    loss_arr = []
    loss_per_epoch = 0
    optimizer = optim.SGD(model.parameters(),0.01,momentum=0.9,weight_decay=1e-4)
    train_loader , _ = get_train_test_loaders()
    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.1)
    for epoch in range(8):
        loss_per_epoch = train(model,loss_func,optimizer,train_loader,epoch)
        loss_arr.append(loss_per_epoch)
        scheduler.step()
    torch.save(model.state_dict(),"checkpoint.pth")
    plot_losses(loss_arr)
    


if __name__ == '__main__':
    main()