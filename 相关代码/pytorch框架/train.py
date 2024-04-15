import torch
import torch.nn as nn
import torch.optim as optim
import torchvision 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
            
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),
        )

        self.fc = nn.Sequential(
            nn.Linear(512*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 3)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1) 
        return self.fc(out)
def train(model, train_loader, criterion, optimizer, scheduler, epoch, num_epochs):
    model.train()

    total_loss = 0
    total_correct = 0
    total_data = 0
    train_bar = tqdm(train_loader)
    for data in train_bar:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        #梯度清零
        optimizer.zero_grad()
        #正向传播
        outputs = model(images)
        _, predicted = torch.max(outputs.data, dim=1)
        total_correct += torch.eq(predicted, labels).sum().item()
        #计算损失
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        #反向传播
        loss.backward()
        #权重更新
        optimizer.step()

        total_data += labels.size(0)
        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch+1, num_epochs, loss)
    
    scheduler.step()
    train_loss = total_loss / len(train_loader)
    train_acc = total_correct / total_data
    print('Train epoch{}: Loss:{:.4f}  Acc:{:.4f}'.format(epoch+1, train_loss, train_acc))


def validate(model, val_loader, criterion, epoch, num_epochs):
    model.eval()
    
    total_loss = 0
    total_correct = 0
    total_data = 0
    with torch.no_grad():
        val_bar = tqdm(val_loader)
        for data in val_bar:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            #正向传播
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total_correct += torch.eq(predicted, labels).sum().item()
            #计算损失
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total_data += labels.size(0)
            val_bar.desc = "test epoch[{}/{}]".format(epoch + 1, num_epochs)
        val_loss = total_loss / len(val_loader)
        val_acc =  total_correct / total_data
        print('Validate epoch{}: Loss:{:.4f}  Acc:{:.4f}'.format(epoch+1, val_loss, val_acc))

if __name__ == '__main__':
    learning_rate = 0.001
    batch_size = 32
    num_epochs = 30

    train_path = r'/home/mist/mask_project/train'
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_data = datasets.ImageFolder(train_path, train_transforms)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    class_name = train_data.classes


    val_path = r'/home/mist/mask_project/val'
    val_trasforms = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_data = datasets.ImageFolder(val_path, val_trasforms)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    print(class_name)
    model = CNN()
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    for epoch in range(num_epochs):
        train(model, train_loader, criterion, optimizer, scheduler, epoch, num_epochs)
        validate(model, val_loader, criterion, epoch, num_epochs)

    torch.save(model.state_dict(),"/home/mist/mask_project/model/mask.pth")
    torch.save(model,"/home/mist/mask_project/model/mask_whole.pth")
