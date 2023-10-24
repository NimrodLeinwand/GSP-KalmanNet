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


nl_L = torch.tensor([[ 3., -1.,  0.,  0.,  0.,  0., -1.,  0.,  0., -1.],
        [-1.,  3.,  0.,  0.,  0.,  0.,  0.,  0., -1., -1.],
        [ 0.,  0.,  3., -1.,  0., -1., -1.,  0.,  0.,  0.],
        [ 0.,  0., -1.,  3., -1.,  0.,  0.,  0.,  0., -1.],
        [ 0.,  0.,  0., -1.,  3., -1.,  0., -1.,  0.,  0.],
        [ 0.,  0., -1.,  0., -1.,  3.,  0.,  0., -1.,  0.],
        [-1.,  0., -1.,  0.,  0.,  0.,  3., -1.,  0.,  0.],
        [ 0.,  0.,  0.,  0., -1.,  0., -1.,  3., -1.,  0.],
        [ 0., -1.,  0.,  0.,  0., -1.,  0., -1.,  3.,  0.],
        [-1., -1.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  3.]])

# nl_L = torch.Tensor([[6, -1, 0, -1, -1, -1, -1, 0, 0, -1],
#                          [-1, 6, 0, -1, 0, -1, -1, -1, 0, -1],
#                          [0, 0, 6, -1, -1, -1, 0, -1, -1, -1],
#                          [-1, -1, -1, 6, -1, 0, -1, 0, 0, -1],
#                          [-1, 0, -1, -1, 6, 0, -1, -1, -1, 0],
#                          [-1, -1, -1, 0, 0, 6, 0, -1, -1, -1],
#                          [-1, -1, 0, -1, -1, 0, 6, -1, -1, 0],
#                          [0, -1, -1, 0, -1, -1, -1, 6, -1, 0],
#                          [0, 0, -1, 0, -1, -1, -1, -1, 6, -1],
#                          [-1, -1, -1, -1, 0, -1, 0, 0, -1, 6]])

epsilon = 0
nl_L = nl_L.type(torch.DoubleTensor)
W, V = np.linalg.eig(nl_L)
nl_V = torch.from_numpy(V).type(torch.DoubleTensor).to(dev)+epsilon
nl_V_t = torch.transpose(nl_V, 0, 1).type(torch.DoubleTensor).to(dev).float()
nl_V = nl_V.float()
nl_T = 200   # In case of other size of trajectory need to be changed
T = nl_T
nl_T_test = nl_T
nl_n = 10    # In case of other size of vector need to be changed
nl_m = nl_n  # In case of other size of vector need to be changed

def nl_f(x):
    return 10*(x/10+torch.sin(x/9+3))


def nl_h(x):
    return 3 * x


def nl_f_EKF(x):
    return 10*(x/10+torch.sin(x/9+3)).to(dev)

def nl_h_EKF(x):
    return 3 * x


def nl_getJacobian(x, a):
    try:
        if (x.size()[1] == 1):
            y = torch.reshape((x.T), [x.size()[0]])
    except:
        y = torch.reshape((x.T), [x.size()[0]])

    if (a == 'ObsAcc'):
        g = nl_h_EKF
    elif (a == 'ModAcc'):
        g = nl_f_EKF

    Jac = autograd.functional.jacobian(g, y)
    Jac = Jac.view(-1, nl_m)
    return Jac.to(dev)

