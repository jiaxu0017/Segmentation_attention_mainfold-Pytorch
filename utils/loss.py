import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


class ManifondLoss(nn.Module):
    def __init__(self, alpha=1, size_average=True, ignore_index=255):
        super(ManifondLoss,self).__init__()
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.size_average = size_average
        # self.coss_manifode

    def forward(self, input, targets):
        ce_loss = self.coss_manifode(input, targets,k = 3)
        return ce_loss * self.alpha

    def coss_manifode(self, inputs, targets,  k=3):
        # print(inputs.shape, targets.shape)
        k_l = int(k/2)
        h = inputs.size(2)
        w = inputs.size(3)
        inputs= inputs.detach().max(dim=1)[1]
        # print(inputs.shape, targets.shape)
        input = inputs[:, k_l:h-k+k_l , k_l:w-k+k_l].float()
        target = targets[:, k_l:h-k+k_l , k_l:w-k+k_l].float()
        # temp = torch.Tensor(input.size())
        temp = torch.zeros(input.size()).cuda()
        for i in range(k):
            for j in range(k):
                output = inputs[:, k_l+i:h-k+i+k_l, k_l+j:w-k+j+k_l].float()
                target_out = targets[:, k_l+i:h-k+i+k_l, k_l+j:w-k+j+k_l].float()

                # print(input.size(),output.size())
                temp += torch.pow((input-output),2) * torch.exp(-torch.pow((target-target_out),2))

        return torch.mean(temp)


if __name__ == '__main__':
    input = torch.ones([8,19,5,5])
    for i  in range(5):
        for j in range(5):
            input[:,:,i,j] = i * 5 + j
    loss = coss_manifode(input)
    # print(input)
    # output = input[:,:,0:-1-2,:0:-1-2]
    # temp = input[:,:,0:5-2,0:5-2] - output[:,:,0:5-2,0+1:5-1]
    print(loss)
    # print(loss)
    # print(temp)
    # print(temp.size())
    # print(loss.size())

    input = torch.ones([2,2,5,5])
    for i  in range(5):
        for j in range(5):
            input[:,:,i,j] = i * 5 + j

    out = torch.pow(input,2)
    # print(out)
