import torch, code,copy
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from cifar_dataset import cifar10 
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
net = Net()


transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]) 
trainset = cifar10('cifar', transform,True)

#dataiter = iter(trainloader)
#images, labels = dataiter.next()
images, labels = trainset[0]
images = images.unsqueeze(0)

### 1. you can pick up the predefined layers
print("input shapes: ", images.shape)
after_conv = net.conv1(images); print("after convolution: ", after_conv.shape)
after_relu = F.relu(after_conv)
after_pool = net.pool(after_relu); print("after pooling: ", after_pool.shape)


### 2. the convolution operation works like that
print(net.state_dict().keys())
conv1_weight = net.state_dict()['conv1.weight']
conv1_bias = net.state_dict()['conv1.bias']

direct_conv = F.conv2d(images,conv1_weight,conv1_bias)

print(direct_conv.shape,"direct_conv")
print(after_conv.shape,"normal_conv")

### 3. how about fully connected layers
fc1_weight = net.state_dict()['fc1.weight']
fc1_bias = net.state_dict()['fc1.bias']

after_conv1 = net.pool(F.relu(net.conv1(images)))
after_conv2 = net.pool(F.relu(net.conv2(after_conv1)))
flattend = after_conv2.view(-1, 16 * 5 * 5)

direct_fc1 = torch.matmul(flattend,fc1_weight.T) + fc1_bias
after_fc1 = net.fc1(flattend)

print(direct_fc1.shape,"direct_fc1")
print(after_fc1.shape,"after_fc1")


### 4. register forward hook, simply capture middle values 
global save_hook
save_hook = {}
def forward_value_saving(self, input, output):
    # input is a tuple of packed inputs
    # output is a Tensor. output.data is the Tensor we are interested

    save_hook["input"] = input
    save_hook["output"] = output
    print('Inside ' + self.__class__.__name__ + ' forward!!!')


net.conv1.register_forward_hook(forward_value_saving)
pred = net(images)

print(save_hook['input'][0].shape,"hook input")
print(save_hook['output'][0].shape,"hook output")

print(save_hook['input'][0].shape,"hook input")
print(save_hook['output'][0].shape,"hook output")



### 5. register backward hook, simply capture middle (gradient) values 
def backward_value_saving(self, input, output):
    # input is a tuple of packed inputs
    # output is a Tensor. output.data is the Tensor we are interested

    save_hook["back_input"] = input
    save_hook["back_output"] = output
    print('Inside ' + self.__class__.__name__ + ' backward!!!')

net.conv1.register_backward_hook(backward_value_saving)
labels = torch.tensor([3], dtype=torch.long)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

pred = net(images) # here the forward hook saved
loss = criterion(pred,labels)
loss.backward() # here the back hook saved
optimizer.step()
code.interact(local = dict(globals(),**locals()))

