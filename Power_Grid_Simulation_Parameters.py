import torch
import numpy as np
from torch import autograd
from pypower.api import case118, makeYbus, case14, case30
from pypower.idx_brch import F_BUS, T_BUS



L14 = [
    [19.4980702055144, -15.2630865231796, 0, 0, -4.23498368233483, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [-15.2630865231796, 30.3547153987791, -4.78186315175772, -5.11583832587208, -5.19392739796971, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, -4.78186315175772, 9.85068012935164, -5.06881697759392, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, -5.11583832587208, -5.06881697759392, 38.3431317384716, -21.5785539816916, 0, -4.78194338179036, 0, -1.79797907152361, 0, 0, 0, 0, 0],
    [-4.23498368233483, -5.19392739796971, 0, -21.5785539816916, 34.9754041144523, -3.96793905245615, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, -3.96793905245615, 17.3407328099191, 0, 0, 0, 0, -4.09407434424044, -3.17596396502940, -6.10275544819312, 0],
    [0, 0, 0, -4.78194338179036, 0, 0, 19.5490059482647, -5.67697984672154, -9.09008271975275, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, -5.67697984672154, 5.67697984672154, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, -1.79797907152361, 0, 0, -9.09008271975275, 0, 24.2825063752679, -10.3653941270609, 0, 0, 0, -3.02905045693060],
    [0, 0, 0, 0, 0, 0, 0, 0, -10.3653941270609, 14.7683378765214, -4.40294374946052, 0, 0, 0],
    [0, 0, 0, 0, 0, -4.09407434424044, 0, 0, 0, -4.40294374946052, 8.49701809370096, 0, 0, 0],
    [0, 0, 0, 0, 0, -3.17596396502940, 0, 0, 0, 0, 0, 5.42793859120161, -2.25197462617221, 0],
    [0, 0, 0, 0, 0, -6.10275544819312, 0, 0, 0, 0, 0, -2.25197462617221, 10.6696935494707, -2.31496347510535],
    [0, 0, 0, 0, 0, 0, 0, 0, -3.02905045693060, 0, 0, 0, -2.31496347510535, 5.34401393203596]
]

PG_L14 = torch.tensor(L14)

const = 15

patrial_L14 = [
    [19.4980702055144 -const, -15.2630865231796 +const, 0, 0, -4.23498368233483, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [-15.2630865231796 +const, 30.3547153987791 -const, -4.78186315175772, -5.11583832587208, -5.19392739796971, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, -4.78186315175772, 9.85068012935164-5, -5.06881697759392+5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, -5.11583832587208, -5.06881697759392+5, 38.3431317384716-5, -21.5785539816916, 0, -4.78194338179036, 0, -1.79797907152361, 0, 0, 0, 0, 0],
    [-4.23498368233483, -5.19392739796971, 0, -21.5785539816916, 34.9754041144523, -3.96793905245615, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, -3.96793905245615, 17.3407328099191, 0, 0, 0, 0, -4.09407434424044, -3.17596396502940, -6.10275544819312, 0],
    [0, 0, 0, -4.78194338179036, 0, 0, 19.5490059482647, -5.67697984672154, -9.09008271975275, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, -5.67697984672154, 5.67697984672154, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, -1.79797907152361, 0, 0, -9.09008271975275, 0, 24.2825063752679, -10.3653941270609, 0, 0, 0, -3.02905045693060],
    [0, 0, 0, 0, 0, 0, 0, 0, -10.3653941270609, 14.7683378765214, -4.40294374946052, 0, 0, 0],
    [0, 0, 0, 0, 0, -4.09407434424044, 0, 0, 0, -4.40294374946052, 8.49701809370096, 0, 0, 0],
    [0, 0, 0, 0, 0, -3.17596396502940, 0, 0, 0, 0, 0, 5.42793859120161, -2.25197462617221, 0],
    [0, 0, 0, 0, 0, -6.10275544819312, 0, 0, 0, 0, 0, -2.25197462617221, 10.6696935494707-2, -2.31496347510535+2],
    [0, 0, 0, 0, 0, 0, 0, 0, -3.02905045693060, 0, 0, 0, -2.31496347510535+2, 5.34401393203596-2]
]

partial_PG_L14 = torch.tensor(patrial_L14)


L = L14
def get_nl_model():
  ppc = case14()
  # ppc = case30()
  branch = ppc["branch"]
  nl = ppc["branch"].shape[0]
  f = branch[:, F_BUS] - np.ones(nl)
  t = branch[:, T_BUS] - np.ones(nl)
  branch[:, F_BUS] = f
  branch[:, T_BUS] = t
  ppc["branch"] = branch
  Ybus, Yf, Yt = makeYbus(ppc["baseMVA"], ppc["bus"], ppc["branch"])
  return Ybus

# simulation setup parameters
Ybus = get_nl_model()
Ybus = Ybus.todense()
Ybus = torch.tensor(Ybus)
# print(Ybus)

###### Partial: ########
Ybus_part = Ybus
mis_Ybus = 1.8+1.8j
Ybus_part[0,0] -= mis_Ybus
Ybus_part[1,0] += mis_Ybus
Ybus_part[0,1] += mis_Ybus
Ybus_part[1,1] -= mis_Ybus
########################

# print(Ybus_part)
# print(L)
# print(L.shape)
nl_n_x = np.shape(Ybus)[0]
nl_n_y = np.shape(Ybus)[0]
n_x = nl_n_x
n_y = nl_n_y
P = 100

pypower_T = P
pypower_T_test = pypower_T



################### Full L #############################
pypower_L = torch.tensor(L)
L = torch.tensor(L)
prior_C_yy = -np.linalg.inv(Ybus.imag[1:, 1:].cpu())
################### Partial L ##########################
# Ybus = Ybus_part
# pypower_L = L30
# L = partial_PG_L14
# prior_C_yy = -np.linalg.inv(Ybus_part.imag[1:, 1:].cpu())

U, S, Vh = np.linalg.svd(prior_C_yy, full_matrices=True)
S = np.diag(S)
sqrt_R_y = np.matmul(np.matmul(U, np.sqrt(S)), Vh)

pypower_m = 14         # For other cases need to be change according the case vector size
pypower_n = pypower_m

W, V = np.linalg.eig(pypower_L.cpu())
pypower_V = torch.from_numpy(V).type(torch.FloatTensor)
pypower_V_t = torch.transpose(pypower_V, 0, 1)

# Get adjacency matrix A from Laplacian matrix L
if torch.cuda.is_available():
    A1 = (torch.diag_embed(torch.diag(L)) - L).cuda.clone().detach().requires_grad_(True)
    A = torch.diag(L)
    A = (A - L).cuda().clone().detach().requires_grad_(True)
else:
    A = torch.diag(L)
    A = (A - L).cpu()

def pypower_f_EKF(x):
    if torch.cuda.is_available():
        x = x.cuda().clone().detach().requires_grad_(True)
    return x.float()+0.05

def pypower_h_EKF(y):
    v = torch.exp(1j * y)
    Ybus_conj_v = torch.matmul(torch.conj(torch.transpose(Ybus, 0, 1)).to(torch.complex64), torch.conj(v).to(torch.complex64))
    s = torch.mul(v, Ybus_conj_v)
    p = torch.real(s)
    return p

def pypower_f(x):
    return x.float()+0.05


def pypower_h(y):
    B = y.size()[0]
    theta = y
    if torch.cuda.is_available():
        v = torch.exp(1j * torch.tensor(theta, dtype=torch.float)).cuda().clone().detach().requires_grad_(True)
    else:
        v = torch.exp(1j * torch.tensor(theta, dtype=torch.float))
    trans = torch.transpose(Ybus, 1, 0)
    Ybus_conj_v = torch.bmm(torch.conj(trans).expand(B,-1,-1).float(), torch.conj(v).unsqueeze(-1).float()).clone().detach().requires_grad_(True)
    s = torch.mul(v.squeeze(), Ybus_conj_v.squeeze())
    p = torch.real(s).clone().detach()
    return p

def pypower_getJacobian(x, a):
    # if(x.size()[1] == 1):
    #     y = torch.reshape((x.T),[x.size()[0]])
    try:
        if (x.size()[1] == 1):
            y = torch.reshape((x.T), [x.size()[0]])
    except:
        y = torch.reshape((x.T), [x.size()[0]])
    if (a == 'ObsAcc'):
        g = pypower_h_EKF
    elif (a == 'ModAcc'):
        g = pypower_f_EKF
    if torch.isnan(x).any():
      print("Variable F contains NaN values. ", g)
    Jac = autograd.functional.jacobian(g, y)
    Jac = Jac.view(-1, pypower_m)
    return Jac