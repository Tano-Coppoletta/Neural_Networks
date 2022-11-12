import torch
import torch.nn as nn
from torch.autograd import Variable 
import numpy as np
import random

def transform_letter_for_name(name):
    matrix_of_letters = np.zeros((11,27))
    if len(name)>=11:
        new_name = name[-10:]
       # print("NAME:",new_name)
    else:
        new_name=name
    for i in range(11):
        if i < len(new_name):
            ascii_number = ord(new_name[i])-96
        else:
            ascii_number = 0 #end of word encoded
        matrix_of_letters[i, ascii_number] = 1
    return matrix_of_letters


class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes 
        self.num_layers = num_layers 
        self.input_size = input_size 
        self.hidden_size = hidden_size 
        self.seq_length = seq_length 

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm
        self.fc_1 =  nn.Linear(hidden_size, 128) 
        self.fc = nn.Linear(128, num_classes) 

        self.relu = nn.ReLU()
    
    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)) #internal state
        
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) 
        hn = hn.view(-1, self.hidden_size) 
        out = self.relu(output)
        out = self.fc_1(out) 
        out = self.relu(out) 
        out = self.fc(out) 
        return out

input_size = 27 #number of features
hidden_size = 64 #number of features in hidden state
num_layers = 1 #number of stacked lstm layers

num_classes = 27 #number of output classes 

lstm1 = LSTM1(num_classes, input_size, hidden_size, num_layers, 11)
lstm1.eval()

lstm1.load_state_dict(torch.load('0702-657811153-Coppoletta.ZZZ'))
#use the model
l=input("Insert a letter:\n")
letter=l
letter_encoded=transform_letter_for_name(letter)
letter_tensor =torch.tensor(letter_encoded, dtype=torch.float).view(1,11,27)
name=letter
num_letters=1
position=1
num_names=0
while num_names<20:
    while True:
        
        output = lstm1.forward(letter_tensor)
        random_number = random.uniform(0,1)
        while random_number>0.7:
            output[:,num_letters-1, np.argmax(output[:,position-1,:].cpu().detach().numpy())]=0
            random_number = random.uniform(0,1)

        letter = np.argmax(output[:,position-1,:].cpu().detach().numpy())
        letter+=96
        letter= chr(letter)
        if(letter=='`'):
            break
        position+=1
        if position>=11:
            position=10
        name+=letter
        letter_encoded=transform_letter_for_name(name)
        letter_tensor= torch.tensor(letter_encoded, dtype=torch.float).view(1,11,27)
        num_letters+=1

    print(name[:],num_letters)
    letter=l
    position=1
    name = letter
    num_letters=1
    num_names+=1
    letter_encoded=transform_letter_for_name(letter)
    letter_tensor =torch.tensor(letter_encoded, dtype=torch.float).view(1,11,27)

 