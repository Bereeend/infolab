from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

class LeNet(nn.Module):
    def __init__(self,n_class=10):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels = 1,
            out_channels = 20,
            kernel_size = 5
        )
        self.conv2 = nn.Conv2d(
            in_channels = 20,
            out_channels = 50,
            kernel_size = 5
        ) 
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, n_class)
    def forward(self, x):
        x = F.relu(self.conv1(x))   # x:[batch_size,1,28,28] => x:[batch_size,20, 24, 24]
        x = F.max_pool2d(x, 2, 2)   # x:[batch_size,20,24,24] => x:[batch_size,20, 12, 12]
        x = F.relu(self.conv2(x))   # x:[batch_size,20,12,12] => x:[batch_size,50, 8, 8]
        x = F.max_pool2d(x, 2, 2)   # x:[batch_size,50,8,8] => x:[batch_size,50, 4, 4]
        x = x.view(-1, 4*4*50)      # x:[batch_size,50,4,4] => x:[batch_size,50*4*4]
        x = F.relu(self.fc1(x))     # x:[batch_size,50*4*4] => x:[batch_size,500]
        x = self.fc2(x)             # x:[batch_size,500] => x:[batch_size,10]
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--filename', type=str, default='sumbission', metavar='FN',
                        help='the filename of the submission')

    filename = input("Please add a descriptive filename: ")
    print("You entered " + str(filename))
    
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = "cuda:0" if use_cuda else "cpu"

    ## Update the 
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_set = dset.MNIST(root='./mnist', train=True, transform=trans)
    test_set = dset.MNIST(root='./mnist', train=False, transform=trans)


    #divide the set to training and validation
    from torch.utils.data.sampler import SubsetRandomSampler
    num_train = len(train_set)
    valid_size = 0.1
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size= args.batch_size,                 
        sampler=train_sampler)

    valid_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size= args.batch_size,                 
        sampler=valid_sampler)
    

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=False)

    print('number of training data:', len(train_set)-split)
    print('number of validation data:', split)
    print('number of test data:', len(test_set))
    
    model = LeNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    #criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch, criterion)
        val(args, model, device, valid_loader, criterion)

    plt.plot(np.arange(len(train_loss)), train_loss)
    plt.show()
    test(args, model, device, test_loader, criterion, filename)
    ## TESTING



def train(args, model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset) - 6000,
                100. * batch_idx / len(train_loader), loss.item()))

def val(args, model, device, val_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target) # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(val_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accura cy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, 6000,
        100. * correct / 6000))        


def test(args, model, device, test_loader, criterion, filename):
    preds = []
    correct_cnt = 0
    total_cnt = 0.0

    for batch_idx, (x, target) in enumerate(test_loader):
        x, target = x.to(device), target.to(device)
        logits = model(x)
        loss = criterion(logits, target)
        _, pred_label = torch.max(logits, 1)
        preds.append(pred_label)
        total_cnt += x.size()[0]
        correct_cnt += (pred_label == target).sum()

        if(batch_idx+1) % 1000 == 0 or (batch_idx+1) == len(test_loader):
            print('==>>> #test_samples: {}, acc: {:.3f}'.format(batch_idx+1, correct_cnt.item() * 1.0 / total_cnt))

     ## Writing to file
    with open(filename + '.csv','w') as file:
        file.write('Id,Label\n')
        for idx, lbl in enumerate(preds): 
            line = '{},{}'.format(idx,lbl.item())
            file.write(line)
            file.write('\n')    


if __name__ == '__main__':
    main()

