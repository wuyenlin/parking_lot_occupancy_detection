#!/usr/bin/python3
from utils.model import *
from utils.dataloader import *

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

parser = argparse.ArgumentParser("Set parking lot occupancy detection project parameters", add_help=False)

parser.add_argument('--epochs', type=int, default=18, help="rounds of training")
parser.add_argument('--imshow', type=bool, default=False, help="show some training dataset")
parser.add_argument('--model', type=str, default='mAlexNet', help='model name')
parser.add_argument('--path', type=str, default='', help='trained model path')
parser.add_argument('--train_img', type=str, default='CNRPark-Patches-150x150/', help="path to training set images")
parser.add_argument('--train_lab', type=str, default='splits/CNRParkAB/even.txt', help="path to training set labels")
parser.add_argument('--test_img', type=str, default='CNRPark-Patches-150x150/', help="path to test set images")
parser.add_argument('--test_lab', type=str, default='splits/CNRParkAB/odd.txt', help="path to test set labels")

parser.add_argument("--device", default="cuda", help="device used")
args = parser.parse_args()



def train(epoch, img_path, target_path, transform, net, criterion, device):
    train_dataset = Data(img_path, target_path, transform)
    train_loader = DataLoader(train_dataset, batch_size=64, \
            shuffle=True, num_workers=0,drop_last=False, collate_fn=collate_fn)
    for ep in range(epoch):  
        if ep >= 12:
            learning_rate = 0.0025
        elif ep >= 6:
            learning_rate = 0.005
        else:
            learning_rate = 0.01
        running_loss = 0.0
        print("Epoch {}.".format(ep+1))
        for i, data in enumerate(train_loader,1):
            inputs, labels = data
            labels = list(map(int, labels))
            labels = torch.Tensor(labels)
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)
            optimizer.zero_grad()
            outputs = net(inputs)

            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            print("Epoch {}.\tBatch {}.\tLoss = {:.3f}.".format(ep+1, i+1, running_loss))
            if i % 2000 == 1999:    # 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    print('Finished Training.')


def test(img_path, target_path, transform, net, device):
    print("\nTesting starts now...")
    test_dataset = Data(img_path, target_path, transform)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=True, \
                            num_workers=0, collate_fn=collate_fn)
    correct = 0
    total = 0
    item = 1
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            print("Testing on batch {}".format(item))
            labels = list(map(int, labels))
            labels = torch.Tensor(labels)
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            item += 1
    return (correct/total)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),  # normalize to [0, 1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]) 
    if args.imshow == True:
        from utils.imshow import imshow
        train_dataset = Data(args.train_img, args.train_lab, transform)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=False)
        imgs, labels = train_loader.__iter__().__next__()
        imshow(train_loader)

    if args.model == 'mAlexNet':
        net = mAlexNet().to(device)
    elif args.model == 'AlexNet':
        net = AlexNet().to(device)

    criterion = nn.CrossEntropyLoss()
    if args.path == '':
        train(args.epochs, args.train_img, args.train_lab, transform, net, criterion, device)
        PATH = './model.pth'
        torch.save(net.state_dict(), PATH)
        net.load_state_dict(torch.load(PATH))

    else:
        PATH = args.path
        net.load_state_dict(torch.load(PATH))
    accuracy = test(args.test_img, args.test_lab, transform, net, device)
    print("\nThe accuracy of training on '{}' and testing on '{}' is {:.3f}.".format(args.train_lab.split('.')[0], args.test_lab.split('.')[0], accuracy))


if __name__=="__main__":
    main()