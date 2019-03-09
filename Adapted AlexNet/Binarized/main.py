import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import model
from torch.autograd import Variable
import os
from logger import Logger

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
best_acc = 0
start_epoch = 0
logger = Logger('./logs')

# Dataset
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
trainset = torchvision.datasets.CIFAR10(root='C:/Users/bryce/Desktop/CIFAR-10', train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=0)
testset = torchvision.datasets.CIFAR10(root='C:/Users/bryce/Desktop/CIFAR-10', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)


# Model
print('==> Building model..')
model = model.AAlexNet()
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def update_lr(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * 0.33

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_index, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs = Variable(inputs)
        targets = Variable(targets)
        # Forward
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # Backward and Optimize
        optimizer.zero_grad()
        loss.backward()
        for p in list(model.parameters()):
            if hasattr(p, 'org'):
                p.data.copy_(p.org)
        optimizer.step()
        for p in list(model.parameters()):
            if hasattr(p, 'org'):
                p.org.copy_(p.data.clamp_(-1, 1))
        '''
        if batch_index % 50 == 0:
            for p in model.parameters():
                print(p.name, p.data)
        '''
        # Results
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    print('Loss: %.3f | Accuracy: %.3f' % (train_loss/128, 100.*correct/total))
    # Update lr
    if epoch == 9 or epoch == 19 or epoch == 29 or epoch == 39 or epoch == 49:
        update_lr(optimizer)


def test(epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_index, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = Variable(inputs)
            targets = Variable(targets)
            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # Results
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    print('Loss: %.3f | Accuracy: %.3f' % (test_loss / 100, 100. * correct / total))

    # Save the model
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/AAlexNet.ckpt')
        best_acc = acc

    # Plot the model
    info = {'loss': test_loss, 'accuracy': acc}
    for tag, value in info.items():
        logger.scalar_summary(tag, value, epoch+1)
    for tag, value in model.named_parameters():
        tag = tag.replace('.', '/')
        logger.histo_summary(tag, value.data.cpu().numpy(), epoch+1)
        logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), epoch+1)

for epoch in range(start_epoch, start_epoch + 80):
    train(epoch)
    test(epoch)