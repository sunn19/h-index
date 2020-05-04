# -*- coding: utf-8 -*-
"""
Created on Mon May  4 15:14:48 2020

@author: Tao Yangtianze & Sun Nan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as Data


###############################################################################

data_1 = pd.read_excel('author_info.xlsx')
data_1.head()
data_1.dropna()
data = np.array(data_1)

###############################################################################
# 数据准备
yau_data = data[:,[1,2,3,4,5,6]][1:45,][:,[0,1,3,4,5]]
shi_data =  data[5:,[1,2,3,4,5,6]][45:58,][:,[0,1,3,4,5]]
rao_data = data[5:,[1,2,3,4,5,6]][58:,][:,[0,1,3,4,5]]

rao_data = np.array([[0, 1, 0, 1, 51],
       [3, 4, 106, 2, 52],
       [7, 8, 162, 3, 53],
       [10, 16, 325, 4, 54],
       [12, 22, 553, 5, 55],
       [14, 30, 822, 6, 56],
       [18, 37, 1243, 7, 57],
       [18, 42, 1523, 8, 58],
       [21, 47, 1991, 9, 59],
       [23, 58, 2202, 10, 60],
       [65, 24, 2399, 11, 61],
       [26, 72, 2540, 12, 62],
       [27, 81, 2653, 13, 63],
       [27, 87, 2664, 14, 64],
       [30, 99, 3222, 15, 65],
       [30, 107, 3318, 16, 66],
       [30, 111, 3339, 17, 67],
       [31, 131, 3573, 18, 68],
       [32, 149, 3674, 19, 69],
       [32, 157, 3678, 20, 70]])

Yan_data = np.array(
       [[1, 1, 2, 1, 41],
       [1, 2, 2, 2, 42],
       [3, 5, 27, 3, 43],
       [3, 6, 27, 4, 44],
       [3, 6, 27, 5, 45],
       [3, 8, 185, 6, 46],
       [5, 10, 237, 7, 47],
       [5, 11, 238, 8, 48],
       [8, 18, 528, 9, 49],
       [10, 21, 2788, 10, 50],
       [13, 27, 3074, 11, 51],
       [15, 29, 3179, 12, 52],
       [17, 34, 3351, 13, 53],
       [19, 36, 3445, 14, 54],
       [22, 41, 4051, 15, 55],
       [27, 53, 4609, 16, 56],
       [28, 61, 4886, 17, 57],
       [29, 63, 5064, 18, 58],
       [30, 67, 5324, 19, 59],
       [32, 77, 5678, 20, 60],
       [35, 88, 5994, 21, 61],
       [35, 105, 6170, 22, 62],
       [35, 113, 6173, 23, 63]])

Bao_data =  np.array([[1, 1, 57, 1, 43],
       [6, 7, 263, 2, 44],
       [6, 7, 263, 3, 45],
       [8, 10, 362, 4, 46],
       [11, 14, 581, 5, 47],
       [14, 22, 1141, 6, 48],
       [15, 24, 1416, 7, 49],
       [19, 32, 2166, 8, 50],
       [21, 39, 2290, 9, 51],
       [25, 51, 3204, 10, 52],
       [28, 59, 4316, 11, 53],
       [29, 63, 4459, 12, 54],
       [32, 72, 5476, 13, 55],
       [35, 82, 6694, 14, 56],
       [39, 101, 8464, 15, 57],
       [41, 107, 9826, 16, 58],
       [48, 136, 11285, 17, 59],
       [50, 159, 11653, 18, 60],
       [50, 182, 11770, 19, 61],
       [50, 189, 11772, 20, 62]])

DH_data =  np.array([[1, 1, 27, 1, 37],
       [2, 2, 36, 2, 38],
       [6, 6, 406, 3, 39],
       [8, 8, 565, 4, 40],
       [12, 13, 858, 5, 41],
       [13, 15, 945, 6, 42],
       [18, 20, 1309, 7, 43],
       [22, 24, 1576, 8, 44],
       [26, 33, 1803, 9, 45],
       [29, 40, 2170, 10, 46],
       [32, 52, 2421, 11, 47],
       [33, 64, 2722, 12, 48],
       [35, 78, 3091, 13, 49],
       [35, 88, 3177, 14, 50],
       [35, 102, 3278, 15, 51],
       [35, 112, 3300, 16, 52],
       [35, 115, 3300, 17, 53]])

T_data =  np.array([[0, 1, 0, 1, 38],
       [2, 3, 88, 2, 39],
       [2, 4, 88, 3, 40],
       [5, 7, 123, 4, 41],
       [7, 15, 194, 5, 42],
       [7, 18, 201, 6, 43],
       [9, 23, 283, 7, 44],
       [10, 27, 615, 8, 45],
       [10, 29, 634, 9, 46],
       [16, 43, 1646, 10, 47],
       [20, 52, 3641, 11, 48],
       [24, 59, 4099, 12, 49],
       [30, 74, 6426, 13, 50],
       [33, 85, 7211, 14, 51],
       [36, 103, 8525, 15, 52],
       [40, 124, 9419, 16, 53],
       [44, 155, 10658, 17, 54],
       [44, 181, 10997, 18, 55],
       [44, 196, 11101, 19, 56],
       [44, 206, 11115, 20, 57]])

LH_data = np.array([[1, 1, 16, 1, 25],
       [1, 1, 16, 2, 26],
       [2, 2, 67, 3, 27],
       [3, 3, 74, 4, 28],
       [5, 6, 108, 5, 29],
       [7, 8, 167, 6, 30],
       [9, 12, 277, 7, 31],
       [13, 19, 390, 8, 32],
       [17, 28, 679, 9, 33],
       [18, 31, 739, 10, 34],
       [20, 36, 967, 11, 35],
       [22, 49, 1243, 12, 36],
       [23, 58, 1359, 13, 37],
       [25, 81, 1755, 14, 38],
       [28, 90, 2065, 15, 39],
       [28, 95, 2156, 16, 40],
       [29, 107, 2405, 17, 41],
       [31, 119, 2607, 18, 42],
       [32, 137, 2962, 19, 43],
       [32, 145, 3001, 20, 44],
       [32, 153, 3052, 21, 45],
       [32, 173, 3086, 22, 46],
       [32, 175, 3086, 23, 47]])



XF_data = np.array([[3, 3, 145, 1, 47],
       [6, 6, 281, 2, 48],
       [13, 14, 605, 3, 49],
       [17, 18, 771, 4, 50],
       [22, 30, 1153, 5, 51],
       [25, 47, 1383, 6, 52],
       [25, 56, 1443, 7, 53],
       [25, 59, 1443, 8, 54]])





WJ_data = np.array([[1, 1, 1, 1, 40],
       [1, 2, 15, 2, 41],
       [2, 3, 44, 3, 42],
       [6, 8, 1990, 4, 43],
       [9, 15, 2114, 5, 44],
       [9, 16, 2256, 6, 45],
       [12, 23, 2722, 7, 46],
       [14, 26, 2799, 8, 47],
       [15, 30, 3129, 9, 48],
       [15, 31, 3140, 10, 49],
       [16, 34, 3173, 11, 50],
       [16, 40, 3174, 12, 51],
       [16, 44, 3193, 13, 52],
       [16, 49, 3195, 14, 53],
       [16, 51, 3195, 15, 54]])
###############################################################################
# data_required_1:出版物总数 data_required_2:被引用频数总计
# 归一化
data = np.array([yau_data,shi_data,rao_data,Yan_data,Bao_data,DH_data,T_data,LH_data,WJ_data])
data_required_1 = []
data_required_2 = []
for x in data:
    x1 = x[:,1][-12:]
    x2 = x[:,3][-12:]
    x1 = (x1-min(x1))/(max(x1)-min(x1))
    x2 = (x2-min(x2))/(max(x2)-min(x2))
    data_required_1.extend(x1)
    data_required_2.extend(x2)


###############################################################################
# 神经网络1
    
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters 
input_size = 1
hidden_size = 4
num_classes = 1
num_epochs = 3
batch_size = 1
learning_rate = 0.001 # Default setting

# Data loader

train_y = np.array(data_required_1).flatten()
train_x = np.array([[1,2,3,4,5,6,7,8,9,10,11,12]*9]).flatten()
train_y = np.array(train_y,dtype = np.float32)
train_x = np.array(train_x,dtype = np.float32)

train_x = torch.from_numpy(train_x)
train_y = torch.from_numpy(train_y)

train_dataset = Data.TensorDataset(train_x,train_y)

train_loader = Data.DataLoader(
    dataset = train_dataset,
    batch_size = batch_size,
    shuffle=False,
)

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model_1 = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model_1.parameters(), lr=learning_rate)  

Loss_1 = []

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (x, y) in enumerate(train_loader):  
        
        # Forward pass
        outputs = model_1(x)
        loss = criterion(outputs, y)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        if (i+1) % 1 == 0:
            Loss_1.append(loss.item())
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
        
# Loss_1 
fig, ax = plt.subplots(figsize=(5, 4))

plt.plot(Loss_1)
plt.xlabel('Iterations')
plt.ylabel('Loss')

plt.tight_layout()
plt.show()

fig.savefig('Loss_1.eps',dpi=600,format='eps',bbox_inches='tight')

###############################################################################
# 神经网络2

# Hyper-parameters 
input_size = 1
hidden_size = 8
num_classes = 1
num_epochs = 2
batch_size = 1
learning_rate = 0.001 # Default setting

# Data loader

train_y = np.array(data_required_2).flatten()
train_x = np.array([[1,2,3,4,5,6,7,8,9,10,11,12]*9]).flatten()
train_y = np.array(train_y,dtype = np.float32)
train_x = np.array(train_x,dtype = np.float32)

train_x = torch.from_numpy(train_x)
train_y = torch.from_numpy(train_y)

train_dataset = Data.TensorDataset(train_x,train_y)

train_loader = Data.DataLoader(
    dataset = train_dataset,
    batch_size = batch_size,
    shuffle=False,
)

# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

model_2 = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model_2.parameters(), lr=learning_rate)  


Loss_2 = []
# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        # Move tensors to the configured device
        #images = images.reshape(-1, 28*28).to(device)
        #labels = labels.to(device)
        
        # Forward pass
        outputs = model_2(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 1 == 0:
            Loss_2.append(loss.item())
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Loss_2

fig, ax = plt.subplots(figsize=(5, 4))

plt.plot(Loss_2)
plt.xlabel('Iterations')
plt.ylabel('Loss')

plt.tight_layout()
plt.show()

fig.savefig('Loss_2.eps',dpi=600,format='eps',bbox_inches='tight')

###############################################################################
# Use the model to predict
 
model_1 = model_1.eval() # converting to test model
model_2 = model_2.eval() # converting to test model


Fu_1 = np.array([1,1,5,7,9,14,16,24,29,35,46,54,60])
Fu_2 = np.array([11,11,704,900,953,1175,1297,1616,2879,3032,3180,3222,3222])
l_1_min = min(Fu_1)
l_2_min = min(Fu_2)
scale_1 = max(Fu_1)-l_1_min
scale_2 = max(Fu_2)-l_2_min

def f(x,scale,min):
    return x*scale+min

### 预测后12年
L_1,L_2 = [],[]

for x in np.linspace(12,23,12):
    x = torch.tensor([x])
    y1 = model_1(x)
    y2 = model_2(x)
    L_1.append(f(y1,scale_1,l_1_min))
    L_2.append(f(y2,scale_2,l_1_min))
    
L_1 = [x.detach() for x in L_1]
L_1 = np.array(L_1)
L_2 = [x.detach() for x in L_2]
L_2 = np.array(L_2)

L_1,L_2 = np.array(L_1),np.array(L_2)
L_3 = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
L_4 = np.linspace(33,44,12)

Fu = np.vstack((L_1,L_2,L_3,L_4)) # 整合数据
Fu = Fu.T

###############################################################################
# LSTM
# data_required_3:科研年龄 data_required_4:实际年龄

### data loader
data = np.array([yau_data,shi_data,rao_data,Yan_data,Bao_data,DH_data,T_data,LH_data,WJ_data])
data_required_1 = []
data_required_2 = []
data_required_3 = []
data_required_4 = []
label = []
for x in data:
    y = x[:,0][-12:]
    x1 = x[:,1][-12:]
    x2 = x[:,2][-12:]
    x3 = x[:,3][-12:]
    x4 = x[:,4][-12:]
    x1 = (x1-min(x1))/(max(x1)-min(x1))
    x2 = (x2-min(x2))/(max(x2)-min(x2))
    x3 = (x3-min(x3))/(max(x3)-min(x3))
    x4 = (x4-min(x4))/(max(x4)-min(x4))
    y = (y-min(y))/(max(y)-min(y))
    data_required_1.extend(x1)
    data_required_2.extend(x2)
    data_required_3.extend(x3)
    data_required_4.extend(x4)
    label.extend(y)
    
train_x  = np.vstack((data_required_1,data_required_2,data_required_3,data_required_4))
train_x = np.array(train_x,dtype = np.float32)
train_y  = np.array(label,dtype = np.float32)

train_x = train_x.T
train_x = torch.from_numpy(train_x)
train_x = train_x.view(train_x.shape[0],1,train_x.shape[1])
train_y = torch.from_numpy(train_y)

### Main

# Hyper-parameters
input_size = 4
hidden_size = 32
num_layers = 2
output_size = 1
batch_size = 4
num_epochs = 10
learning_rate = 0.001

train_dataset = Data.TensorDataset(train_x,train_y)

train_loader = Data.DataLoader(
    dataset = train_dataset,
    batch_size = batch_size,
    shuffle=False,
)

# Recurrent neural network
class lstm(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,num_layers):
        super(lstm,self).__init__()
        self.layer1 = nn.LSTM(input_size,hidden_size,num_layers,batch_first=True)
        self.layer2 = nn.Linear(hidden_size,output_size)
    
    def forward(self,x):
        x,_ = self.layer1(x)
        s,b,h = x.size()
        x = x.view(s*b,h)
        x = self.layer2(x)
        x = x.view(s,b,-1)
        return x

model = lstm(input_size, hidden_size, output_size, num_layers)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

Loss_3 = []
# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i,(x,y) in enumerate(train_loader):
        var_x = x
        var_y = y
        # forward pass
        out = model(var_x)
        loss = criterion(out, var_y)
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 6 == 0:
            Loss_3.append(loss.item())
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                       .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
            
# Loss_3
fig, ax = plt.subplots(figsize=(5, 4))

plt.plot(Loss_3)
plt.xlabel('Iterations')
plt.ylabel('Loss')

plt.tight_layout()
plt.show()

fig.savefig('LSTM.eps',dpi=600,format='eps',bbox_inches='tight')

###############################################################################
# Use LSTM to predict h-index
        
model = model.eval() # converting to test model

z1,z2,z3,z4 = Fu[:,0],Fu[:,1],Fu[:,2],Fu[:,3]
z1 = (z1-min(z1))/(max(z1)-min(z1))
z2 = (z2-min(z2))/(max(z2)-min(z2))
z3 = (z3-min(z3))/(max(z3)-min(z3))
z4 = (z4-min(z4))/(max(z4)-min(z4))
Fu[:,0],Fu[:,1],Fu[:,2],Fu[:,3] = z1,z2,z3,z4

h_prediction = []
for i in range(Fu.shape[0]):
    x = Fu[i]
    x = np.array(x,dtype = np.float32)
    x = torch.tensor(x).view(1,1,4)
    y = model(x)
    h_prediction.append(y)

h_prediction = np.array([x.detach() for x in h_prediction],dtype = np.float32)
h_prediction = h_prediction/h_prediction[0]

minx = 1
scale = 20-1
def g(x,scale,minx):
    return x*scale+minx

h = [g(x,scale,minx) for x in h_prediction]

print(h)

x = np.linspace(2021,2032,12)
plt.plot(x,h)
plt.xlabel('Time')
plt.ylabel('h-index')




