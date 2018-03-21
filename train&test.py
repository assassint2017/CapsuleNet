from time import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

from CapsuleNet import CapsuleNet

# Hyper parameters
epoch_num = 300
batch_size = 64
lr = 0.0001  # Adam default learning rate

dataset_dir = './MNIST/'  # the path of your dataset
img_channel = 1
img_size = 28
num_class = 10  # class number of your data

primarycaps_dim = 8  # the dim of Primary Capsule
digitcaps_dim = 16  # the dim of Digit Capsule
iter_num = 3  # the iteration time of routing

workers = 1  # subprocess number for load the data

test_ds_size = 10000  # the size of your test dataset


# dataset
train_ds = torchvision.datasets.MNIST(dataset_dir, transform=transforms.ToTensor())
train_dl = DataLoader(train_ds, batch_size, True, num_workers=workers)

test_ds = torchvision.datasets.MNIST(dataset_dir, False, transforms.ToTensor())
test_dl = DataLoader(test_ds, batch_size, num_workers=workers)


# use cuda if you have GPU
net = CapsuleNet(img_channel, num_class, img_size, primarycaps_dim, digitcaps_dim, iter_num).cuda()


# optimizer
opt = torch.optim.Adam(net.parameters(), lr=lr)  # optimizer for network


# loss function
reconstruction_loss_func = nn.MSELoss()


def one_hot(label, expand=False):  # convert the GT label to onehot code

    one_hot_code = torch.zeros((label.size(0), num_class))

    for i in range(label.size(0)):

        one_hot_code[i][int(label[i])] = 1

    if expand is True:  # return the expand onehot code

        one_hot_code = torch.unsqueeze(one_hot_code, 2)

        return Variable(one_hot_code.expand(label.size(0), num_class, digitcaps_dim)).cuda()

    else:

        return Variable(one_hot_code).cuda()


# train the network
start = time()
number = 1
figure = plt.figure('Visualization')

for epoch in range(epoch_num):

    for step, (test_data, label) in enumerate(train_dl, 1):

        test_data, label = test_data.cuda(), label.cuda()

        tc = one_hot(label)

        clf, reconstruction = net(test_data, True, one_hot(label, True))

        reconstruction_loss = reconstruction_loss_func(reconstruction, test_data)

        margin_loss = torch.mean(torch.sum(tc * F.relu(0.9 - clf) ** 2 + 0.5 * (1 - tc) * F.relu(clf - 0.1) ** 2, dim=1))

        loss = margin_loss + 0.0005 * reconstruction_loss  # scale down the reconstruction loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 40 is 0:

            # Visualization of the input image and the reconstruction image

            figure.clf()

            figure.add_subplot(121)
            train_img = torchvision.utils.make_grid(test_data.data.cpu())
            plt.title('input_img')
            plt.imshow(train_img.permute(1, 2, 0).numpy())

            figure.add_subplot(122)
            reconstruction_img = torchvision.utils.make_grid(reconstruction.data.cpu())
            plt.title('reconstruction_img')
            plt.imshow(reconstruction_img.permute(1, 2, 0).numpy())

            figure.savefig('./img/' + str(number) + '.png')

            number += 1
            plt.pause(0.01)

            # test for acc of both train and test dataset
            test_acc, train_acc = 0, 0

            for (test_data, label) in test_dl:

                test_data, label = test_data.cuda(), label.cuda()

                clf = net(test_data)

                test_acc += sum(torch.max(clf, 1)[1].data.cpu().numpy() == label.data.cpu().numpy())

            test_acc /= test_ds_size

            print('epoch:{}, step:{}, test_acc:{:.3f} %, time:{:.1f} min'
                  .format(epoch, step, test_acc * 100, (time() - start) / 60))

