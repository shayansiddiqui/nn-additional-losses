"""
Addition losses module defines classses which are commonly used particularly in segmentation and are not part of standard pytorch library.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss, _WeightedLoss
import numpy as np
from torch.autograd import Variable

class DiceLossBinary(_Loss):
    """
    Dice Loss for a batch of samples
    """

    def forward(self, output, target):
        """
        Forward pass

        :param output: Nx1xHxW Variable
        :param target: NxHxW LongTensor
        :return:
        """

        eps = 0.0001

        intersection = output * target
        numerator = 2 * intersection.sum(0).sum(1).sum(1)
        denominator = output + target
        denominator = denominator.sum(0).sum(1).sum(1) + eps
        loss_per_channel = 1 - (numerator / denominator)

        return loss_per_channel.sum() / output.size(1)


class DiceCoeff(nn.Module):
    """
    Dice coeff for individual examples
    """

    def __init__(self):
        super(DiceCoeff, self).__init__()

    def forward(self, input, target):
        """
        Forward pass

        :param input: torch.tensor (CxHxW)
        :param target:
        :return: float scaler
        """
        inter = torch.dot(input, target)
        union = torch.sum(input ** 2) + torch.sum(target ** 2) + 0.0001

        t = 2 * inter.float() / union.float()
        return t

class IoULoss(_WeightedLoss):
    """
    Dice Loss for a batch of samples
    """

    def forward(self, output, target, weights=None, ignore_index=None):
        """
        Forward pass

        :param output: NxCxHxW Variable
        :param target: NxHxW LongTensor
        :param weights: C FloatTensor
        :param ignore_index: int index to ignore from loss
        :return:
        """

        eps = 0.0001
        encoded_target = output.detach() * 0

        if ignore_index is not None:
            mask = target == ignore_index
            target = target.clone()
            target[mask] = 0
            encoded_target.scatter_(1, target.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(encoded_target)
            encoded_target[mask] = 0
        else:
            encoded_target.scatter_(1, target.unsqueeze(1), 1)

        if weights is None:
            weights = 1

        intersection = output * encoded_target
        numerator = intersection.sum(0).sum(1).sum(1)
        denominator = (output + encoded_target) - (output*encoded_target)

        if ignore_index is not None:
            denominator[mask] = 0
        denominator = denominator.sum(0).sum(1).sum(1) + eps
        loss_per_channel = weights * (1 - (numerator / denominator))

        return loss_per_channel.sum() / output.size(1)



class DiceLoss(_WeightedLoss):
    """
    Dice Loss for a batch of samples
    """

    def forward(self, output, target, weights=None, ignore_index=None):
        """
        Forward pass

        :param output: NxCxHxW Variable
        :param target: NxHxW LongTensor
        :param weights: C FloatTensor
        :param ignore_index: int index to ignore from loss
        :return:
        """

        eps = 0.0001
        encoded_target = output.detach() * 0

        if ignore_index is not None:
            mask = target == ignore_index
            target = target.clone()
            target[mask] = 0
            encoded_target.scatter_(1, target.unsqueeze(1), 1)
            mask = mask.unsqueeze(1).expand_as(encoded_target)
            encoded_target[mask] = 0
        else:
            encoded_target.scatter_(1, target.unsqueeze(1), 1)

        if weights is None:
            weights = 1

        intersection = output * encoded_target
        numerator = 2 * intersection.sum(0).sum(1).sum(1)
        denominator = output + encoded_target

        if ignore_index is not None:
            denominator[mask] = 0
        denominator = denominator.sum(0).sum(1).sum(1) + eps
        loss_per_channel = weights * (1 - (numerator / denominator))

        return loss_per_channel.sum() / output.size(1)


class CrossEntropyLoss2d(_WeightedLoss):
    """
    Standard pytorch weighted nn.CrossEntropyLoss
    """

    def __init__(self, weight=None):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.CrossEntropyLoss(weight)

    def forward(self, inputs, targets):
        """
        Forward pass

        :param inputs: torch.tensor (NxC)
        :param targets: torch.tensor (N)
        :return: scalar
        """
        return self.nll_loss(inputs, targets)


class CombinedLoss(_Loss):
    """
    A combination of dice  and cross entropy loss
    """

    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.cross_entropy_loss = CrossEntropyLoss2d()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss()
        self.l2_loss = nn.MSELoss()

    def forward(self, input, target, weight=None, grad_weight=True):
        """
        Forward pass

        :param input: torch.tensor (NxCxHxW)
        :param target: torch.tensor (NxHxW)
        :param weight: torch.tensor (NxHxW)
        :return: scalar
        """
        # TODO: why?
        # TODO: Need to discuss this
        # input_soft = F.softmax(input, dim=1)
        # y2 = torch.mean(self.dice_loss(input_soft, target))
        # if weight is None:
        #     y1 = torch.mean(self.cross_entropy_loss.forward(input, target))
        # else:
        #     y1 = torch.mean(torch.mul(self.cross_entropy_loss.forward(input, target), weight.cuda()))
        # y = y1 + y2
        # return y
        input_soft = F.softmax(input, dim=1)
        y1 = torch.mean(self.dice_loss(input_soft, target))
        y2 = self.focal_loss(input, target)
        if weight is not None:
            w1, w2 = weight
            return w1*y1 + w2*y2
        else:
            return y1+y2
        # y = y1 + y2
        # if weight is None and grad_weight == False:
        #    y1 = torch.mean(self.cross_entropy_loss.forward(input, target))
        # else:
        #    if grad_weight == True:
        #        filters = np.array([[-1, 1, -1], [0, 0, 0], [1, -1, 1]])
        #        directional_filter = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        #
        #        directional_filter.weight.data.copy_(torch.from_numpy(filters))
        #        # directional_filter.bias.data.copy_(torch.from_numpy(np.zeros(shape=(filters.shape,))))
        #
        #        # weight = torch.tensor(5 * np.gradient(target.cpu().detach().numpy())).type(torch.FloatTensor).cuda()
        #        weight = directional_filter(target.unsqueeze(dim=1).type(torch.FloatTensor))
        #        weight = (weight != 0).type(torch.FloatTensor)
        #    y1 = torch.mean(torch.mul(self.cross_entropy_loss.forward(input, target), 2*weight.cuda()))
        # y = 0.001*y1 + y2
        # _, out = torch.max(input_soft, dim=1)
        # y2 = self.l2_loss(out.type(torch.cuda.FloatTensor), target.type(torch.cuda.FloatTensor))

class DiceL2Loss(_Loss):
    """
    A combination of dice  and L2 loss of projected area
    """

    def __init__(self):
        super(DiceL2Loss, self).__init__()
        self.l2_loss = nn.MSELoss()
        self.dice_loss = DiceLoss()

    def forward(self, input, target, cond_target=None):
        """
        Forward pass

        :param input: torch.tensor (NxCxHxW)
        :param target: torch.tensor (NxHxW)
        :param cond_target: torch.tensor (NxHxW)
        :return: scalar
        """

        input_soft = F.softmax(input, dim=1)
        y1 = torch.mean(self.dice_loss(input_soft, target))
        _, out = torch.max(input_soft, dim=1)
        # out, target, cond_target = out.type(torch.cuda.FloatTensor), target.type(torch.cuda.FloatTensor), cond_target.type(torch.cuda.FloatTensor)
        y2 = self.l2_loss(out.type(torch.cuda.FloatTensor), target.type(torch.cuda.FloatTensor))
        # batch, _, _ = target.size()
        # actual_area_diff = torch.abs(torch.mean(target.view(batch, -1), dim=1) - torch.mean(cond_target.view(batch, -1), dim=1))
        # _, out = torch.max(input, dim=1)
        # estimated_area_diff = torch.abs(torch.mean(out.view(batch, -1), dim=1) - torch.mean(cond_target.view(batch, -1), dim=1))
        # y2 = self.l2_loss(estimated_area_diff.type(torch.cuda.FloatTensor), actual_area_diff.type(torch.cuda.FloatTensor))
        # y = y1 + (0.0001 * y2)
        y = y1 + y2
        return y


# Copied from https://github.com/clcarwin/focal_loss_pytorch
class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()