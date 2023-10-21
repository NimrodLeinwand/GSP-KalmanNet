import torch
import torch.nn as nn
import torch.nn.functional as func
from Main_Pypower import dev

nGRU = 2
class ExtendedKalmanNetNN(torch.nn.Module):

    ###################
    ### Constructor ###
    ###################
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ######################################
    ### Initialize Kalman Gain Network ###
    ######################################

    def Build(self, ssModel, infoString='fullInfo'):

        self.InitSystemDynamics(ssModel.f, ssModel.h, ssModel.m, ssModel.n, infoString='fullInfo')
        self.InitSequence(ssModel.m1x_0, ssModel.T)

        # Number of neurons in the 1st hidden layer
        H1_KNet = (ssModel.m + ssModel.n) * 8
        # Number of neurons in the 2nd hidden layer
        H2_KNet = (ssModel.m * ssModel.n)  * 1 * 2

        self.InitKGainNet(H1_KNet, H2_KNet)

    def InitKGainNet(self, H1, H2):
        # Input Dimensions (+1 for time input)
        D_in = self.m + self.m + self.n  # F1,3,4
        # Output Dimensions
        D_out = self.m * self.n  # Kalman Gain

        ###################
        ### Input Layer ###
        ###################
        # Linear Layer
        self.KG_l1 = torch.nn.Linear(D_in, H1, bias=True) #.type(torch.DoubleTensor)
        # ReLU (Rectified Linear Unit) Activation Function
        self.KG_relu1 = torch.nn.ReLU()
        ###########
        ### GRU ###
        ###########

        # Input Dimension
        self.input_dim = H1
        # Hidden Dimension
        self.hidden_dim = ((self.n * self.n) + (self.m * self.m)) * 8
        # Number of Layers
        self.n_layers = nGRU
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
        self.hn = torch.randn(self.seq_len_hidden, self.batch_size, self.hidden_dim)

        # Iniatialize GRU Layer
        self.rnn_GRU = nn.GRU(self.input_dim, self.hidden_dim, self.n_layers, dropout=0)

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
    def InitSystemDynamics(self, f, h, m, n, infoString='fullInfo'):

        if (infoString == 'partialInfo'):
            self.fString = 'ModInacc'
            self.hString = 'ObsInacc'
        else:
            self.fString = 'ModAcc'
            self.hString = 'ObsAcc'

        # Set State Evolution Function
        self.f = f
        self.m = m

        # Set Observation Function
        self.h = h
        self.n = n

    ###########################
    ### Initialize Sequence ###
    ###########################
    def InitSequence(self, M1_0, T):

        self.m1x_posterior = torch.squeeze(M1_0).to(dev)
        self.m1x_posterior_previous = 0  # for t=0

        self.T = T
        self.x_out = torch.empty(self.m, T)

        self.state_process_posterior_0 = torch.squeeze(M1_0).to(dev)
        self.m1x_prior_previous = self.m1x_posterior

        # KGain saving
        self.i = 0
        self.KGain_array = self.KG_array = torch.zeros((self.T*1000, self.m, self.n)).to(dev)

    ######################
    ### Compute Priors ###
    ######################
    def step_prior(self):
        # Predict the 1-st moment of x
        self.m1x_prior = torch.squeeze(self.f(self.m1x_posterior))

        # Predict the 1-st moment of y
        self.m1y = self.h(self.m1x_prior)


        self.state_process_prior_0 = torch.squeeze(self.f(self.state_process_posterior_0))
        self.obs_process_0 = torch.unsqueeze(self.h(self.state_process_prior_0),-1)

    ##############################
    ### Kalman Gain Estimation ###
    ##############################
    def step_KGain_est(self, y):
        # Feature 1: yt - yt-1
        try:
            my_f1_0 = y - torch.squeeze(self.y_previous.to(dev)).to(dev)
        except:
            my_f1_0 = y - self.obs_process_0.to(dev).to(dev)  # when t=0
        y_f1_norm = func.normalize(my_f1_0, p=2, dim=0, eps=1e-12, out=None)

        self.m1x_posterior = self.m1x_posterior.reshape(-1,self.m,1)
        m1x_f3_0 = self.m1x_posterior - self.m1x_posterior_previous
        m1x_f3_reshape = torch.squeeze(m1x_f3_0)
        m1x_f3_norm = func.normalize(m1x_f3_reshape, p=2, dim=0, eps=1e-12, out=None)
        m1x_f3_norm = m1x_f3_norm.unsqueeze(-1)


        m1x_f4_0 = self.m1x_posterior.squeeze() - self.m1x_prior_previous
        m1x_f4_reshape = torch.squeeze(m1x_f4_0)
        m1x_f4_norm = func.normalize(m1x_f4_reshape, p=2, dim=0, eps=1e-12, out=None)
        m1x_f4_norm = m1x_f4_norm.unsqueeze(-1)


        # Input for counting
        count_norm = func.normalize(torch.tensor([self.i]).float(), dim=0, eps=1e-12, out=None)
        # KGain Net Input
        KGainNet_in = torch.cat([y_f1_norm.to(dev), m1x_f3_norm.to(dev), m1x_f4_norm.to(dev)], dim=1).type(torch.FloatTensor)

        KGainNet_in = KGainNet_in.to(dev)
        # Kalman Gain Network Step
        KG = self.KGain_step(KGainNet_in.to(dev)).to(dev)
        # Reshape Kalman Gain to a Matrix
        self.KGain = torch.reshape(KG, (-1, self.m, self.n)).type(torch.FloatTensor).to(dev)

    #######################
    ### Kalman Net Step ###
    #######################
    def KNet_step(self, y):
        # Compute Priors
        self.step_prior()

        # Compute Kalman Gain
        self.step_KGain_est(y.to(dev))
        # Save KGain in array
        self.KGain_array[self.i:self.i+self.KGain.shape[0]] = self.KGain.to(dev)
        self.i += 1

        # Innovation
        dy = torch.squeeze(y).to(dev) - self.m1y.to(dev)

        # Compute the 1-st posterior moment
        INOV = torch.matmul(self.KGain.float().to(dev), dy.unsqueeze(-1).float())
        self.m1x_posterior_previous = self.m1x_posterior.to(dev)
        self.m1x_posterior = self.m1x_prior.unsqueeze(-1).to(dev) +INOV.to(dev)

        self.state_process_posterior_0 = self.state_process_prior_0.to(dev)
        self.m1x_prior_previous = self.m1x_prior.to(dev)
        self.y_previous = y.to(dev)

        # return
        return torch.squeeze(self.m1x_posterior)

    ########################
    ### Kalman Gain Step ###
    ########################
    def KGain_step(self, KGainNet_in):

        def expand_dim(x):
            expanded = torch.empty(self.seq_len_input, self.batch_size, x.shape[-1]).to(self.device)
            expanded[0, :, :] = x
            return expanded

        ###################
        ### Input Layer ###
        ###################
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)
        # Reshape KGainNet_in to match the input shape expected by KG_l1
        KGainNet_in = KGainNet_in.view(KGainNet_in.size(0), -1)
        L1_out = self.KG_l1(KGainNet_in)
        La1_out = self.KG_relu1(L1_out)

        ###########
        ### GRU ###
        ###########
        GRU_in = torch.zeros(self.seq_len_input, self.batch_size, self.input_dim)
        GRU_in[0, :, :] = La1_out.to(dev)
        GRU_out, self.hn = self.rnn_GRU(GRU_in.type(torch.FloatTensor).to(dev), self.hn.type(torch.FloatTensor).to(dev))

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
    def forward(self, y):
        # yt = torch.squeeze(y)
        if True in torch.isnan(y):
          print(" ####################### Obsrevations contains nan!!")
        '''
        for t in range(0, self.T):
            self.x_out[:, t] = self.KNet_step(y[:, t])
        '''
        self.x_out = self.KNet_step(y.to(dev))
        return self.x_out

    #########################
    ### Init Hidden State ###
    #########################

    def init_hidden(self):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, self.batch_size, self.hidden_dim).zero_()
        self.hn = hidden.data