import torch
import numpy as np
from torch import autograd
PFandUKF_test = False
if torch.cuda.is_available() and not PFandUKF_test:
    dev = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print("Running on the GPU")
else:
    dev = torch.device("cpu")
    torch.set_default_tensor_type('torch.FloatTensor')
    print("Running on the CPU")

r2 = 0.21
vdB = -20  # ratio v=q2/r2
v = 10 ** (vdB / 10)
q2 = torch.mul(v, r2)
q = torch.sqrt(q2)

T = 200
T_test = T
m = 10
n = 10
F = torch.eye(10).to(dev)
H = torch.diag(torch.tensor([0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.4])).to(dev)
m1x_0 = torch.ones(m, 1).to(dev)
m2x_0 = 0 * 0 * torch.eye(m).to(dev)



L = torch.Tensor([[3, -1,  0,  0,  0,  0, -1,  0,  0, -1],
                  [-1, 2, 0,  0,  0,  0,  0,  0, -1,  0],
                  [0, 0,  2, -1,  0,  0,  0, -1,  0,  0],
                  [0,  0, -1,  3, -1, -1,  0,  0,  0,  0],
                  [0,  0,  0, -1,  3,  0,  0, -1, -1,  0],
                  [0,  0,  0, -1,  0,  3,  0,  0, -1, -1],
                  [-1,  0,  0,  0,  0,  0,  3, -1,  0, -1],
                  [0,  0, -1,  0, -1,  0, -1,  3,  0,  0],
                  [0, -1,  0,  0, -1, -1,  0,  0,  3,  0],
                  [-1,  0,  0,  0,  0, -1, -1,  0,  0,  3]]).to(dev)

# L = torch.Tensor([[6, -1, 0, -1, -1, -1, -1, 0, 0, -1],
#                   [-1, 6, 0, -1, 0, -1, -1, -1, 0, -1],
#                   [0, 0, 6, -1, -1, -1, 0, -1, -1, -1],
#                   [-1, -1, -1, 6, -1, 0, -1, 0, 0, -1],
#                   [-1, 0, -1, -1, 6, 0, -1, -1, -1, 0],
#                   [-1, -1, -1, 0, 0, 6, 0, -1, -1, -1],
#                   [-1, -1, 0, -1, -1, 0, 6, -1, -1, 0],
#                   [0, -1, -1, 0, -1, -1, -1, 6, -1, 0],
#                   [0, 0, -1, 0, -1, -1, -1, -1, 6, -1],
#                   [-1, -1, -1, -1, 0, -1, 0, 0, -1, 6]])



W, V = np.linalg.eig(L.cpu())
V = torch.from_numpy(V).type(torch.FloatTensor).to(dev)
V_t = torch.transpose(V, 0, 1).to(dev)
L = torch.tensor(L).type(torch.FloatTensor).to(dev)



torch.pi = torch.acos(torch.zeros(1)).item() * 2  # which is 3.1415927410125732

def f(x):
    return torch.matmul(F.type(torch.DoubleTensor), x.type(torch.DoubleTensor))

def h(x):
    return torch.matmul(H.type(torch.DoubleTensor), x.type(torch.DoubleTensor))

def Naive(x):
    H_inv = torch.inverse(H.to(dev))
    return torch.matmul(H_inv.type(torch.DoubleTensor), x.type(torch.DoubleTensor)).to(dev)

def getJacobian(x, a):
    try:
        if (x.size()[1] == 1):
            y = torch.reshape((x.T), [x.size()[0]])
    except:
        y = torch.reshape((x.T), [x.size()[0]])

    if (a == 'ObsAcc'):
        g = h
    elif (a == 'ModAcc'):
        g = f
    print(x.shape)
    Jac = autograd.functional.jacobian(g, y)
    Jac = Jac.view(-1, m)

    return Jac.to(dev)
