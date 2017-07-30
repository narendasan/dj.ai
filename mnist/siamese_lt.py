import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import os

BATCH_SIZE = 64
TEST_BATCH_SIZE = 1000
EPOCHS = 2
LEARNING_RATE = 0.001
SGD_MOMENTUM = 0.5  
SEED = 1
LOG_INTERVAL = 10

#Enable Cuda
torch.cuda.manual_seed(SEED)

#Dataloader
kwargs = {'num_workers': 1, 'pin_memory': True}
train_loader  = torch.utils.data.DataLoader(
    datasets.MNIST('/tmp/mnist/data', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])),
    batch_size=BATCH_SIZE,
    shuffle=True,
    **kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('/tmp/mnist/data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])),
    batch_size=TEST_BATCH_SIZE,
    shuffle=True,
    **kwargs
    )

#Network
class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(800, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward_twin(self, x):
        x = F.max_pool2d(self.conv1(x), kernel_size=2, stride=2)
        x = F.max_pool2d(self.conv2(x), kernel_size=2, stride=2)
        x = x.view(-1, 800)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def forward(self, in1, in2):
        out1 = self.forward_twin(in1)
        out2 = self.forward_twin(in2)
        return out1, out2


class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, out1, out2, label):
        # out1 and out2 are 64 x 1 x 28 x 28
        out1 = out1.view(BATCH_SIZE, -1).type(torch.FloatTensor).cuda()
        out2 = out2.view(BATCH_SIZE, -1).type(torch.FloatTensor).cuda()
        dist = F.pairwise_distance(out1, out2, p=1)
        # print(type(label))
        # print(type(dist))
        # dist = dist.view(BATCH_SIZE)
        # var = Variable(torch.pow(torch.clamp(self.margin * torch.ones(BATCH_SIZE).type(torch.FloatTensor).cuda() - dist, min = 0.0), 2))
        # loss = torch.mean( (1 + -1* label) * (torch.pow(dist, 2)) +
        #                   (label) *var)
        shift = torch.ones(BATCH_SIZE).type(torch.FloatTensor).cuda()
        label_tensor = label.data.type(torch.cuda.FloatTensor)
        dist_tensor = dist.data.type(torch.cuda.FloatTensor)
        # print(label_tensor.size())
        # print(dist_tensor.size())
        # print(torch.pow(dist_tensor, 2).type())
        loss = torch.add(-1*label_tensor, shift) * torch.pow(dist_tensor, 2).type(torch.cuda.FloatTensor) 
        # print(loss.type())
        # loss = torch.mean(torch.add(loss, torch.mm(label_tensor, torch.pow(torch.clamp(self.margin - dist_tensor, min = 0.0).type(torch.cuda.FloatTensor), 2).type(torch.cuda.FloatTensor)) )).type(torch.cuda.FloatTensor)
        # print(torch.pow(dist_tensor -self.margin*shift, 2).type())
        loss = torch.mean(torch.add(loss, label_tensor * torch.pow(dist_tensor - self.margin*shift, 2))) 
        print(type(loss))
        return Variable(loss, requires_grad = True) 
        
model = SiameseNet()
model.cuda()

crit = ContrastiveLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=SGD_MOMENTUM)


def train(epoch):
    model.train()
    for batch1, (data1, target1) in enumerate(train_loader):
        for batch2, (data2, target2) in enumerate(train_loader):
            data1, data2, target1, target2 = data1.cuda(), data2.cuda(), target1.cuda(), target2.cuda()
            target = torch.lt(target1, target2)
            data1, data2, target = Variable(data1), Variable(data2), Variable(target)
            optimizer.zero_grad()
            out1, out2 = model(data1, data2)
            loss = crit(data1, data2, target)
            print('got here')
            loss.backward()
            print('oh rip')
            optimizer.step()
            if batch % LOG_INTERVAL == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                               batch * len(data),
                                                                               len(train_loader.dataset),
                                                                               100. * batch / len(train_loader), loss.data[0]))

def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    for data1, target1 in test_loader:
        for data2, target2 in test_loader:
            data1, target1 = data1.cuda(), target1.cuda()
            data2, target2 = data2.cuda(), target2.cuda()
            target = torch.lt(target1, target2)
            data1, data2, target = Variable(data1, volatile=True), Variable(data2, volatile=True), Variable(target)
            out1, out2 = model(data1, data2)
            test_loss += crit(out1, out2, target).data[0]
            pred = out1.data.max(1)[1] < out2.data.max(1)[1]
            correct += pred.eq(target.data).cpu().sum()
    test_loss /= len(test_loader)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss,
                                                                                 correct,
                                                                                 len(test_loader.dataset),
                                                                                 100. * correct / len(test_loader.dataset)))

def learn():
    for e in range(EPOCHS):
        train(e + 1)
        test(e + 1)

if __name__ == '__main__':
    learn()
    print(model)
    torch.save(model.state_dict(), os.path.dirname(os.path.realpath(__file__)) + "/trained_mnist.pyt")
