import torch
import torch.nn as nn
import random
import torch.nn.functional as func
from Main_Pypower import dev
class KalmanNetNN(torch.nn.Module):

    ###################
    ### Constructor ###
    ###################
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #############
    ### Build ###
    #############
    def Build(self, ssModel):

        self.InitSystemDynamics(ssModel.F, ssModel.H)

        # Number of neurons in the 1st hidden layer
        # H1_KNet = (ssModel.m + ssModel.n) * (10) * 8
        H1_KNet = (ssModel.m + ssModel.n) * 8

        # Number of neurons in the 2nd hidden layer
        H2_KNet = (ssModel.m * ssModel.n) * 1 * (4)

        self.InitKGainNet(H1_KNet, H2_KNet)

    ######################################
    ### Initialize Kalman Gain Network ###
    ######################################
    def InitKGainNet(self, H1, H2):
        # Input Dimensions
        D_in = self.m + self.n  # x(t-1), y(t)

        # Output Dimensions
        D_out = self.m * self.n  # Kalman Gain

        ###################
        ### Input Layer ###
        ###################
        # Linear Layer
        self.KG_l1 = torch.nn.Linear(D_in, H1, bias=True)

        # ReLU (Rectified Linear Unit) Activation Function
        self.KG_relu1 = torch.nn.ReLU()

        ###########
        ### GRU ###
        ###########
        # Input Dimension
        self.input_dim = H1
        # Hidden Dimension
        self.hidden_dim = (self.m * self.m + self.n * self.n) * 10
        # Number of Layers
        self.n_layers = 1
        # Batch Size
        self.batch_size = 1
        # Input Sequence Length
        self.seq_len_input = 1
        # Hidden Sequence Length
        self.seq_len_hidden = self.n_layers

        # batch_first = False
        # dropout = 0.1 ;

        # Initialize a Tensor for GRU Input
        # self.GRU_in = torch.empty(self.seq_len_input, self.batch_size, self.input_dim)

        # Initialize a Tensor for Hidden State
        self.hn = torch.randn(self.seq_len_hidden, self.batch_size, self.hidden_dim).to(self.device,non_blocking = True)

        # Iniatialize GRU Layer
        self.rnn_GRU = nn.GRU(self.input_dim, self.hidden_dim, self.n_layers)

        ####################
        ### Hidden Layer ###
        ####################
        self.KG_l2 = torch.nn.Linear(self.hidden_dim, H2, bias=True)

        # ReLU (Rectified Linear Unit) Activation Function
        self.KG_relu2 = torch.nn.ReLU()

        ####################
        ### Output Layer ###
        ####################
        self.KG_l3 = torch.nn.Linear(H2, D_out, bias=True)

    ##################################
    ### Initialize System Dynamics ###
    ##################################
    def InitSystemDynamics(self, F, H):
        # Set State Evolution Matrix
        self.F = F.to(self.device,non_blocking = True)
        self.F_T = torch.transpose(F, 0, 1)
        self.m = self.F.size()[0]

        # Set Observation Matrix
        self.H = H.to(self.device,non_blocking = True)
        self.H_T = torch.transpose(H, 0, 1)
        self.n = self.H.size()[0]

    ###########################
    ### Initialize Sequence ###
    ###########################
    # def InitSequence(self, M1_0):
    #     self.m1x_prior = M1_0.to(self.device,non_blocking = True)

    #     self.m1x_posterior = M1_0.to(self.device,non_blocking = True)

    #     self.state_process_posterior_0 = M1_0.to(self.device,non_blocking = True)

    def InitSequence(self, M1_0, T):

        self.m1x_posterior = torch.squeeze(M1_0).to(dev)
        self.m1x_posterior_previous = 0  # for t=0

        self.T = T
        self.x_out = torch.empty(self.m, T)

        self.m1x_prior = M1_0.to(self.device,non_blocking = True)
        self.state_process_posterior_0 = torch.squeeze(M1_0).to(dev)
        self.m1x_prior_previous = self.m1x_posterior

        # KGain saving
        self.i = 0
        self.KGain_array = self.KG_array = torch.zeros((self.T*10, self.m, self.n)).to(dev)

    ######################
    ### Compute Priors ###
    ######################
    def step_prior(self):
        # Compute the 1-st moment of x based on model knowledge and without process noise
        bmm_mul = torch.bmm(self.F.expand(self.state_process_posterior_0.size()[0], -1, -1).type(torch.DoubleTensor).to(dev), self.state_process_posterior_0.unsqueeze(-1).type(torch.DoubleTensor).to(dev)).squeeze(-1)
        self.state_process_prior_0 = bmm_mul

        # Compute the 1-st moment of y based on model knowledge and without noise
        bmm_mul = torch.bmm(self.H.expand(self.state_process_prior_0.size()[0], -1, -1).type(torch.DoubleTensor).to(dev), self.state_process_prior_0.unsqueeze(-1).type(torch.DoubleTensor).to(dev)).squeeze(-1)
        self.obs_process_0 = bmm_mul

        # Predict the 1-st moment of x
        self.m1x_prev_prior = self.m1x_prior.squeeze()
        bmm_mul = torch.bmm(self.F.expand(self.m1x_posterior.size()[0], -1, -1).type(torch.DoubleTensor).to(dev), self.m1x_posterior.unsqueeze(-1).type(torch.DoubleTensor).to(dev)).squeeze(-1)
        self.m1x_prior = bmm_mul

        # Predict the 1-st moment of y
        bmm_mul = torch.bmm(self.H.expand(self.m1x_prior.size()[0], -1, -1).type(torch.DoubleTensor).to(dev), self.m1x_prior.unsqueeze(-1).type(torch.DoubleTensor).to(dev)).squeeze(-1)
        self.m1y = bmm_mul


    ##############################
    ### Kalman Gain Estimation ###
    ##############################
    def step_KGain_est(self, y):

        # Reshape and Normalize the difference in X prior
        # Featture 4: x_t|t - x_t|t-1
        #dm1x = self.m1x_prior - self.state_process_prior_0
        dm1x = self.m1x_posterior - self.m1x_prev_prior
        dm1x_reshape = torch.squeeze(dm1x)
        dm1x_norm = func.normalize(dm1x_reshape, p=2, dim=0, eps=1e-12, out=None)

        # Feature 2: yt - y_t+1|t
        dm1y = y.squeeze() - torch.squeeze(self.m1y)
        dm1y_norm = func.normalize(dm1y, p=2, dim=0, eps=1e-12, out=None)

        # KGain Net Input
        KGainNet_in = torch.cat([dm1y_norm, dm1x_norm], dim=1)

        # Kalman Gain Network Step
        KG = self.KGain_step(KGainNet_in)
        # Reshape Kalman Gain to a Matrix
        self.KGain = torch.reshape(KG, (-1,self.m, self.n))

    #######################
    ### Kalman Net Step ###
    #######################
    def KNet_step(self, y):
        # Compute Priors
        self.step_prior()

        # Compute Kalman Gain
        self.step_KGain_est(y)

        # Innovation
        y_obs = torch.squeeze(y)
        dy = y_obs - self.m1y
        # Compute the 1-st posterior moment
        # bmm_mul = torch.bmm(self.F.expand(self.state_process_posterior_0.size()[0], -1, -1).type(torch.DoubleTensor).to(dev), self.state_process_posterior_0.unsqueeze(-1).type(torch.DoubleTensor).to(dev)).squeeze(-1)
        INOV = torch.matmul(self.KGain.float(), dy.unsqueeze(-1).float()).squeeze()
        self.m1x_posterior = self.m1x_prior + INOV

        # return
        return torch.squeeze(self.m1x_posterior)

    ########################
    ### Kalman Gain Step ###
    ########################
    def KGain_step(self, KGainNet_in):

        ###################
        ### Input Layer ###
        ###################
        L1_out = self.KG_l1(KGainNet_in.type(torch.FloatTensor).to(dev))
        La1_out = self.KG_relu1(L1_out)

        ###########
        ### GRU ###
        ###########
        GRU_in = torch.empty(self.seq_len_input, self.batch_size, self.input_dim).to(self.device,non_blocking = True)
        GRU_in[0, :, :] = La1_out
        GRU_out, self.hn = self.rnn_GRU(GRU_in, self.hn)
        # GRU_out_reshape = torch.reshape(GRU_out, (1, self.hidden_dim))

        ####################
        ### Hidden Layer ###
        ####################
        L2_out = self.KG_l2(GRU_out)
        La2_out = self.KG_relu2(L2_out)

        ####################
        ### Output Layer ###
        ####################
        L3_out = self.KG_l3(La2_out)
        return L3_out

    ###############
    ### Forward ###
    ###############
    def forward(self, yt):
        yt = yt.to(self.device,non_blocking = True)
        return self.KNet_step(yt)

    #########################
    ### Init Hidden State ###
    #########################
    def init_hidden(self):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, self.batch_size, self.hidden_dim).zero_()
        self.hn = hidden.data
