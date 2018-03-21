import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CapsuleNet(nn.Module):
    def __init__(self, img_channel, num_class, img_size, primarycaps_dim, digitcaps_dim, iter_num):
        """

        :param img_channel:
        :param num_class:
        :param img_size:

        :param primarycaps_dim: the dim of Primary Capsule
        :param digitcaps_dim: the dim of Digit Capsule
        :param iter_num: the iteration time of routing
        """
        super().__init__()

        self.num_class = num_class

        self.img_size = img_size

        self.img_channel = img_channel

        self.iter_num = iter_num

        self.digitcaps_dim = digitcaps_dim

        self.conv1 = nn.Sequential(
            nn.Conv2d(img_channel, 256, 9),
            nn.ReLU()
        )

        self.primarycaps = nn.ModuleList(
            [nn.Conv2d(256, 32, 9, 2) for _ in range(primarycaps_dim)]
        )

        # the weight matrix
        self.routing_weights = nn.Parameter(torch.randn(32 * 6 * 6, num_class, digitcaps_dim, primarycaps_dim))

        self.decoder = nn.Sequential(  # decoder is to reconstruction the input image

            nn.Linear(num_class * digitcaps_dim, 512),
            nn.ReLU(),

            nn.Linear(512, 1024),
            nn.ReLU(),

            nn.Linear(1024, img_size ** 2),
            nn.Sigmoid()
        )

    def squashing(self, capsule):  # squashing primary and digit capsule layer

        capsule_square = torch.sum(capsule ** 2, dim=2, keepdim=True)

        scaling_factor = capsule_square / (1 + capsule_square)

        return scaling_factor * (capsule / torch.sqrt(capsule_square + 1e-7))

    def dynamic_routing(self, primary_outputs, batch_size):  # dynamic Routing algorithm

        u_ij = torch.stack([primary_outputs for _ in range(self.num_class)], dim=2).unsqueeze(4)

        b_ij = Variable(torch.zeros(batch_size, 1152, self.num_class, 1, 1)).cuda()

        for _ in range(self.iter_num):

            c_ij = F.softmax(b_ij, dim=2)

            s_ij = torch.matmul(self.routing_weights, u_ij) * c_ij

            v_j = torch.sum(s_ij, dim=1)

            v_j = self.squashing(v_j)

            if _ is not self.iter_num - 1:

                v_j = torch.stack([v_j for __ in range(1152)], dim=1)

                b_ij = b_ij + torch.sum(v_j * s_ij, dim=3, keepdim=True)

        return v_j.squeeze()

    def forward(self, inputs, train=False, onehot_label=None):
        """

        :return: clf: (batch_size, num_class)
                 reconstruction_img: (batch_size, img_channel, img_size, img_size)
        """
        conv_outputs = self.conv1(inputs)

        primary_outputs = []
        for conv_layer in self.primarycaps:

            primary_outputs.append(conv_layer(conv_outputs).view(inputs.size(0), -1, 1))

        primary_outputs = self.squashing(torch.cat(primary_outputs, dim=2))

        digitcaps_outputs = self.dynamic_routing(primary_outputs, inputs.size(0))

        clf = torch.sqrt(torch.sum(digitcaps_outputs ** 2, dim=2))  # the final clf result, same as CNN

        if train is True:

            reconstruction_img = self.decoder((digitcaps_outputs * onehot_label).view(-1, self.num_class * self.digitcaps_dim))
            reconstruction_img = reconstruction_img.view(-1, self.img_channel, self.img_size, self.img_size)

            return clf, reconstruction_img

        else:

            return clf

