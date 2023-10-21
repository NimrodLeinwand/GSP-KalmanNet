import torch
# from Main_Pypower import dev
import time
from Non_Linear_Parameters import nl_getJacobian
from Power_Grid_Simulation_Parameters import pypower_getJacobian

epsilon = 0

if torch.cuda.is_available():
    dev = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print("Running on the GPU")
else:
    dev = torch.device("cpu")
    torch.set_default_tensor_type('torch.FloatTensor')
    print("Running on the CPU")

class ExtendedKalmanFilter:

    def __init__(self, SystemModel, equation=13, model='regular', mode='full'):
        self.L = SystemModel.L.to(dev)
        self.V = SystemModel.V.to(dev)
        self.V_t = SystemModel.V_t.to(dev)
        self.f = SystemModel.f
        self.m = SystemModel.m
        self.r = SystemModel.r

        # Has to be transformed because of EKF non-linearity
        self.Q = SystemModel.Q
        self.Q_gsp = torch.matmul(self.V_t.to(dev), self.Q.to(dev))
        self.Q_gsp = torch.matmul(self.Q_gsp.to(dev), self.V.to(dev)).to(dev)

        self.h = SystemModel.h
        self.n = SystemModel.n

        # Has to be transofrmed because of EKF non-linearity
        self.R = SystemModel.R
        self.R_gsp = torch.matmul(self.V_t.to(dev), self.R.to(dev))  # add
        self.R_gsp = torch.matmul(self.R_gsp.to(dev), self.V).to(dev)   # add

        self.T = SystemModel.T
        self.T_test = SystemModel.T_test

        # Pre allocate KG array
        self.KG_array = torch.zeros((self.T_test, self.m, self.n)).to(dev)
        self.model = model
        self.equation = equation
        # Full knowledge about the model or partial? (Should be made more elegant)
        if (mode == 'full'):
            self.fString = 'ModAcc'
            self.hString = 'ObsAcc'
        elif (mode == 'partial'):
            self.fString = 'ModInacc'
            self.hString = 'ObsInacc'

    # Predict
    def Predict(self):
        # Predict the 1-st moment of x
        self.m1x_prior = torch.matmul(self.V_t.float().to(dev), torch.squeeze(self.f(torch.matmul(self.V.float().to(dev), self.m1x_posterior.float().to(dev))).float()).to(dev))  # X_hat t|t-1 (equation 21)
        # Compute the Jacobians
        if self.model == 'NonLinear':
            self.UpdateJacobians(nl_getJacobian(torch.matmul(self.V, self.m1x_posterior.float().to(dev)), self.fString),
                                 nl_getJacobian(torch.matmul(self.V, self.m1x_prior.float().to(dev)), self.hString))
        else:  # Pypower test
            self.UpdateJacobians(pypower_getJacobian(torch.matmul(self.V, self.m1x_posterior.float().to(dev)), self.fString).to(dev),
                                 pypower_getJacobian(torch.matmul(self.V, self.m1x_prior.float().to(dev)), self.hString).to(dev))


        # Predict the 2-nd moment of x
        self.m2x_prior = torch.matmul(self.F.to(dev), self.m2x_posterior.to(dev)).to(dev)
        self.m2x_prior = torch.matmul(self.m2x_prior.to(dev), self.F_T.to(dev)) + self.Q_gsp   # Pt|t-1 (equation 22)

        # Predict the 1-st moment of y
        temp = self.h(torch.matmul(self.V, self.m1x_prior.float()).float()).float()
        self.m1y = torch.squeeze(torch.matmul(self.V_t, torch.squeeze(temp.to(dev))))  # calc for equation 23
        # Predict the 2-nd moment of y
        self.m2y = torch.matmul(self.H.to(dev), self.m2x_prior)
        self.m2y = torch.matmul(self.m2y, self.H_T.to(dev)) + self.R_gsp                               # S (13 in article) calc for KG equation 20

    # Compute the Kalman Gain
    def KGain(self):
        if self.equation == 13:
            start = time.time()
            self.KG = torch.matmul(self.m2x_prior, self.H_T.to(dev))
            self.KG = torch.matmul(self.KG, torch.inverse(self.m2y+epsilon))    # equation 13
        else:
            start = time.time()
            self.KG = torch.diag(torch.diagonal(torch.matmul(self.m2x_prior, self.H_T.to(dev))))
            self.KG = torch.matmul(self.KG,torch.inverse(epsilon+torch.diag(torch.diagonal(self.m2y))))    # equation 20

        # Save KalmanGain
        self.KG_array[self.i] = self.KG
        self.i += 1

    # Innovation
    def Innovation(self, y):
        self.dy = y - self.m1y

    # Compute Posterior
    def Correct(self):
        # Compute the 1-st posterior moment
        self.m1x_posterior = self.m1x_prior + torch.matmul(self.KG.to(dev), self.dy.to(dev)).to(dev)   # X_hat_t|t (equation 23)

        # Compute the 2-nd posterior moment
        I = torch.eye(torch.matmul(self.KG.to(dev), self.H.to(dev)).size()[1]).to(dev) - torch.matmul(self.KG.to(dev), self.H.to(dev)).to(dev)     # I - KG*H
        self.m2x_posterior = torch.matmul(I, self.m2x_prior.to(dev)).to(dev)                                      # (I - KG*H)*P_t|t-1
        self.m2x_posterior = torch.matmul(self.m2x_posterior.to(dev), torch.transpose(I, 0, 1)).to(dev)            # (I - KG*H)* P_t|t-1 * (I - KG*H)^T  (equation 24)
        self.m2x_posterior = self.m2x_posterior.to(dev) + torch.matmul(self.KG.to(dev), torch.matmul(self.R_gsp.to(dev), torch.transpose(self.KG.to(dev), 0, 1).to(dev)).to(dev))

    def Update(self, y):
        self.Predict()
        self.KGain()
        self.Innovation(y)
        self.Correct()

        return self.m1x_posterior, self.m2x_posterior

    def InitSequence(self, m1x_0, m2x_0):
        self.m1x_0 = torch.matmul(self.V_t.type(torch.DoubleTensor), m1x_0.type(torch.DoubleTensor))
        self.m2x_0 = self.GFT(m2x_0)

        #########################

    def UpdateJacobians(self, F1, H1):
        self.F = self.GFT(F1)
        self.F_T = torch.transpose(self.F, 0, 1)
        # print('H1', H1.shape)
        self.H = self.GFT(H1)
        self.H_T = torch.transpose(self.H, 0, 1)
        if torch.isnan(self.F).any():
            print("Variable F contains NaN values.")
        if torch.isnan(self.H).any():
            print("Variable H contains NaN values.")

    ### Generate Sequence ###
    #########################
    def GenerateSequence(self, y, T):
        # Pre allocate an array for predicted state and variance
        self.x = torch.zeros(size=[self.m, T])
        self.sigma = torch.zeros(size=[self.m, self.m, T])
        # Pre allocate KG array
        self.KG_array = torch.zeros((T, self.m, self.n))
        self.i = 0  # Index for KG_array alocation

        self.m1x_posterior = torch.squeeze(self.m1x_0)
        self.m2x_posterior = torch.squeeze(self.m2x_0)

        for t in range(0, T):
            yt = torch.squeeze(torch.matmul(self.V_t, y[:, t].float()).to(dev))
            xt, sigmat = self.Update(yt)
            self.x[:, t] = torch.squeeze(torch.matmul(self.V, xt))
            self.sigma[:, :, t] = torch.squeeze(sigmat)

    def GFT(self, x):
        return torch.matmul(self.V_t,torch.matmul(x.to(dev), self.V))

    def IGFT(self, x):
        return torch.matmul(self.V, torch.matmul(x.to(dev), self.V_t))