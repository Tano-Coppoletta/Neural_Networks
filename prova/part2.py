import os
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

name = []
image = []
image_path = './images' #current directory, where the images are
for i in os.listdir(image_path):
    name.append(i)
    image.append(Image.open(os.path.join(image_path, i)))
image = list(map(lambda i: transforms.Compose([transforms.ToTensor()])(i).unsqueeze(0),image))

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
def main():
    net = Net()
    net.load_state_dict(torch.load('0602-657811153-Coppoletta.pt'))
    net.eval()
    names = ['Circle','Heptagon','Hexagon','Nonagon','Octagon','Pentagon','Square','Star','Triangle']
    solutions = list(zip(name, list(map(lambda k: names[k], list(map(lambda l: net(l).argmax().item(), image))))))
    textfile = open("prediction.txt", "w")
    for i in solutions:
        print(f'{i[0]}: {i[1]}')
        textfile.write(f'{i[0]}: {i[1]}')
        textfile.write("\n")
    textfile.close()
main()
