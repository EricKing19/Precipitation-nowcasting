import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import numpy as np


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, ignore_index=-100, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        if isinstance(alpha, np.ndarray):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average
        self.ignore_index = ignore_index

    def forward(self, input, target):
        target_clone = target.clone()

        if self.ignore_index != -100:
            target_clone[target == self.ignore_index] = 0

        target_clone = target_clone.reshape([target_clone.shape[0], 1, target_clone.shape[1], target_clone.shape[2]])

        logpt = F.log_softmax(input, dim=1)
        logpt_ = logpt.gather(1, target_clone)
        pt = logpt_.exp()

        loginput = (1-pt)**self.gamma * logpt

        if self.alpha is not None and self.alpha.shape == loginput.shape:
            loginput = self.alpha * loginput
            return F.nll_loss(loginput, target, ignore_index=self.ignore_index)
        else:
            return F.nll_loss(loginput, target, weight=self.alpha, ignore_index=self.ignore_index)


class LDAMLoss(nn.Module):

    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        # what do you want to do?
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(target.shape[0], 1, target.shape[1], target.shape[2]), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(index_float.transpose(1, 2).transpose(2, 3), self.m_list[:, None])
        batch_m = batch_m.transpose(2, 3).transpose(1, 2)
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.weight)


if __name__ == '__main__':
    x = torch.rand(2, 3, 4, 4) * random.randint(1, 10)
    l = torch.tensor(np.random.randint(0, 3, size=32)).long().reshape([2, 4, 4])
    alpha = torch.rand(2, 3, 4, 4) * random.randint(0, 1)

    # output0 = nn.CrossEntropyLoss(ignore_index=3)(x, l)
    # output0 = FocalLoss(gamma=2)(x, l)
    x = x.cuda()
    l = l.cuda()
    output1 = LDAMLoss([100, 100, 100])(x, l)
    # print(output0)
    print(output1)
