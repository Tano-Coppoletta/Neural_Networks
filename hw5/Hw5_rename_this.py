import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import os
import shutil

from torchvision.datasets import ImageFolder

names = ["Circle", "Square", "Octagon", "Heptagon", "Nonagon", "Star", "Hexagon", "Pentagon", "Tringle"]
train_loss=[]
train_acc=[]
test_loss=[]
test_acc=[]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.max_pool1 = nn.MaxPool2d(4, 4)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.max_pool2 = nn.MaxPool2d(5, 5)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.max_pool3 = nn.MaxPool2d(5, 5)
        self.lin1 = nn.Linear(32, 128)
        self.lin2 = nn.Linear(128, 84)
        self.lin3 = nn.Linear(84, 9)
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.max_pool3(x)
        x = torch.flatten(x, 1)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.relu(x)
        x = self.lin3(x)
        return x

#create directories
os.mkdir("images")
os.mkdir("./images/test")
os.mkdir("./images/train")

for i in range (0,len(names)):
  os.mkdir("./images/test/"+names[i])
  os.mkdir("./images/train/"+names[i])


i=0
j=0
names.sort()
filenames = os.listdir("./geometry_dataset/output")
filenames.sort()
for filename in filenames:
  if i < 8000:
    shutil.copy("./geometry_dataset/output/"+filename,"./images/train/"+names[j]+"/")
    i+=1
  elif i<10000:
    shutil.copy("./geometry_dataset/output/"+filename,"./images/test/"+names[j]+"/")
    i+=1
  else:
    j+=1
    i=0
    if j==9:
      break



def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    tot_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()

        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

        tot_loss = tot_loss + loss.item()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.2f}%'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), tot_loss / (batch_idx + 1),
                       100.0 * correct / ((batch_idx + 1) * args.batch_size)))

    print('End of Epoch: {}'.format(epoch))
    train_loss.append(tot_loss / (len(train_loader)))
    train_acc.append(100.0 * correct / (len(train_loader) * args.batch_size))
    print('Training Loss: {:.6f}, Training Accuracy: {:.2f}%'.format(
        tot_loss / (len(train_loader)), 100.0 * correct / (len(train_loader) * args.batch_size)))


def test(args, model, device, test_loader):
    model.eval()
    tot_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            tot_loss += torch.nn.CrossEntropyLoss()(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss.append(tot_loss / (len(test_loader)))
    test_acc.append(100.0 * correct / (len(test_loader) * args.test_batch_size))
    print('Test Loss: {:.6f}, Test Accuracy: {:.2f}%'.format(
        tot_loss / (len(test_loader)), 100.0 * correct / (len(test_loader) * args.test_batch_size)))


def main():
    # Training settings

    parser = argparse.ArgumentParser(description='HW5')
    parser.add_argument('--batch-size', type=int, default=100, help='input batch size for training (default: 100)')
    parser.add_argument('--test-batch-size', type=int, default=100,
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--gamma', type=float, default=0.7, help='Learning rate step gamma')
    parser.add_argument('--seed', type=int, default=4499)
    parser.add_argument('--log-interval', type=int, default=100,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True, help='For Saving the current Model')
    parser.add_argument('-f')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor()])

    dataset1 = datasets.ImageFolder('./images/train',transform)
    dataset2 = datasets.ImageFolder('./images/test',transform)
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=100, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=100)

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, 20):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "/content/drive/MyDrive/Colab Notebooks/0602-657811153-Coppoletta.pt")

    plt.plot(train_loss, c = 'g')
    plt.plot(test_loss, c = 'r')
    plt.legend(['Train Loss','Test Loss'])
    plt.title("Train Loss & Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig('/content/drive/MyDrive/Colab Notebooks/loss', format = 'eps')
    plt.show()
    plt.plot(train_acc, c = 'g')
    plt.plot(test_acc, c = 'r')
    plt.legend(['Train Accuracy','Test Accuracy'])
    plt.title("Train Accuracy & Test Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.savefig('/content/drive/MyDrive/Colab Notebooks/accuracy', format = 'eps')
    plt.show()



if __name__ == '__main__':
    main()
