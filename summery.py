import torch
from model import myNet1
import torch.nn as nn
from myconfig import myParser

def ttt(args):
    torch.autograd.set_detect_anomaly(True)
    net = myNet1(args)

    in_res = 1536
    Batch = 1

    x = torch.randn(Batch,3,in_res,in_res)

    y = torch.randn(Batch,1,in_res,in_res)
    loss_func = nn.MSELoss(True)

    net.cuda()
    x = x.cuda()
    y = y.cuda()

    # net.eval()

    z = net(x)
    loss = loss_func(z[0],y)
    loss.backward()


    print(z)

if __name__ == '__main__':

    args = myParser()

    ttt(args)

