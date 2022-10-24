import torch

from DiceLoss import *
from BceLoss import *
from torch import nn

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self,y_p,y):
        bceloss = BCELoss2d()
        Diceloss = SoftDiceLoss()
        return 0.6 * bceloss(y_p,y) + 0.4 * Diceloss(y_p,y)


if __name__ == '__main__':
    m = torch.rand(10,10)
    n = torch.rand(10,10)

    l = Loss()
    print(l(m,n))