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

class BeStNet(nn.Module):
    def __init__(self,n_class=10):
        super(BeStNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels = 1,
            out_channels = 64,
            kernel_size = 3
        )
        self.conv2 = nn.Conv2d(
            in_channels = 64,
            out_channels = 128,
            kernel_size = 3
        )
        self.conv3 = nn.Conv2d(
            in_channels = 128,
            out_channels = 256,
            kernel_size = 3
        )
        self.fc1 = nn.Linear(9*256, 500)
        self.fc2 = nn.Linear(500, n_class)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)   # x:[batch_size,20,24,24] => x:[batch_size,20, 12, 12]
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = x.view(-1, 9*256)     # x:[batch_size,50,4,4] => x:[batch_size,50*4*4]
        x = F.relu(self.fc1(x))     # x:[batch_size,50*4*4] => x:[batch_size,500]
        x = self.fc2(x)             # x:[batch_size,500] => x:[batch_size,10]
        return x

train_loss_tot = []

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
    parser.add_argument('--weightdecay', type=float, default=1e-4, metavar='wd',
                        help='SGD weightdecay (default: 1e-4)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--filename', type=str, default='sumbission', metavar='FN',
                        help='the filename of the submission')

    ##Input for what the csv output file should be named
    filename = input("Please add a descriptive filename: ")
    print("You entered " + str(filename))
    
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = "cuda:0" if use_cuda else "cpu"

    ## Update the 

    ## Use different transformations during training and testing.
    trans_train = transforms.Compose([transforms.RandomRotation(degrees=(-8, 8)),
                                transforms.RandomResizedCrop(size=28, scale=(0.95, 1.05)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,),(0.3081,))])

    ## Images should not be rotated and cropped in the test set
    trans_test = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,),(0.3081,))])

    train_set = dset.MNIST(root='./mnist', train=True, transform=trans_train)
    test_set = dset.MNIST(root='./mnist', train=False, transform=trans_test)


    #divide the set to training and validation
    from torch.utils.data.sampler import SubsetRandomSampler
    num_train = len(train_set)
    valid_size = 0.015
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
    
    model = BeStNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay = args.weightdecay)
    #criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(args, optimizer, epoch)
        train(args, model, device, train_loader, optimizer, epoch, criterion)
        val(args, model, device, valid_loader, criterion)

    plt.plot(np.arange(len(train_loss_tot)), train_loss_tot)
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
        train_loss_tot.append(loss)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset) - 900,
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
    print('\nValdidation set: Average loss: {:.4f}, Accura cy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, 900,
        100. * correct / 900))        


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
            print('==>>> #test_samples: {}, acc: {:.7f}'.format(batch_idx+1, correct_cnt.item() * 1.0 / total_cnt))

     ## Writing to file
    with open(filename + '.csv','w') as file:
        file.write('Id,Label\n')
        for idx, lbl in enumerate(preds): 
            line = '{},{}'.format(idx,lbl.item())
            file.write(line)
            file.write('\n')    

def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = args.lr * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr            

if __name__ == '__main__':
    main()

