"""# **Class: KalmanNet**"""
import torch
import torch.nn as nn
import torch.nn.functional as func

if torch.cuda.is_available():
    dev = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print("Running on the GPU")
else:
    dev = torch.device("cpu")
    torch.set_default_tensor_type('torch.FloatTensor')
    print("Running on the CPU")

nGRU = 2
class GSPExtendedKalmanNetNN(torch.nn.Module):

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
        self.V = ssModel.V
        self.V_t = ssModel.V_t
        self.InitSystemDynamics(ssModel.f, ssModel.h, ssModel.m, ssModel.n, infoString='fullInfo')

        # Number of neurons in the 1st hidden layer
        H1_KNet = (ssModel.m + ssModel.n) * 8

        # Number of neurons in the 2nd hidden layer
        H2_KNet = (ssModel.n) * 1 * (4)

        self.InitKGainNet(H1_KNet, H2_KNet)

    def InitKGainNet(self, H1, H2):
        # Input Dimensions (+1 for time input)
        D_in = self.m + self.m + self.n  # F1,3,4

        # Output Dimensions
        D_out = self.n  # Diagonal Kalman Gain

        ###################
        ### Input Layer ###
        ###################

        # Linear Layer
        self.KG_l1 = torch.nn.Linear(D_in, H1, bias=True).type(torch.FloatTensor)

        # ReLU (Rectified Linear Unit) Activation Function
        self.KG_relu1 = torch.nn.ReLU()

        ###########
        ### GRU ###
        ###########

        # Input Dimension
        self.input_dim = H1
        # Hidden Dimension
        # self.hidden_dim = ((self.n * self.n) + (self.m * self.m)) * 10 * 1
        self.hidden_dim = ((self.n + self.m)) * 10
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
        M1_0 = self.GFT(M1_0.type(torch.FloatTensor).to(dev))
        self.m1x_posterior = M1_0.unsqueeze(-1)
        self.m1x_posterior_previous = 0  # for t=0

        self.T = T
        self.x_out = torch.zeros(self.m, T)

        self.state_process_posterior_0 = torch.squeeze(M1_0)
        self.m1x_prior_previous = self.m1x_posterior

        # KGain saving
        self.i = 0
        self.KGain_array = self.KG_array = torch.zeros((self.T*20, self.m, self.n)).to(dev)

    ######################
    ### Compute Priors ###
    ######################
    def step_prior(self):
        # Predict the 1-st moment of x
        temp = self.f(self.BMM_multipy(self.V.type(torch.DoubleTensor), self.m1x_posterior.type(torch.DoubleTensor))).type(torch.DoubleTensor)
        self.m1x_prior = self.BMM_multipy(self.V_t, torch.squeeze(temp).type(torch.FloatTensor).to(dev))

        # Predict the 1-st moment of y
        temp = torch.squeeze(self.h(self.BMM_multipy(self.V, self.m1x_prior)))
        # print(temp.shape)
        self.m1y = self.BMM_multipy(self.V_t, temp)

        # Update Jacobians
        # self.JFt = get_Jacobian(self.m1x_posterior, self.fString)
        # self.JHt = get_Jacobian(self.m1x_prior, self.hString)

        self.state_process_prior_0 = torch.squeeze(self.f(self.state_process_posterior_0))
        self.obs_process_0 = torch.squeeze(self.h(self.state_process_prior_0))

    ##############################
    ### Kalman Gain Estimation ###
    ##############################
    def step_KGain_est(self, y):
        # Feature 1: yt - yt-1
        try:
            my_f1_0 = y.to(dev) - torch.squeeze(self.y_previous).to(dev)
        except:
            my_f1_0 = y.to(dev) - torch.squeeze(self.obs_process_0).to(dev)  # when t=0
        # my_f1_reshape = torch.squeeze(my_f1_0)
        y_f1_norm = func.normalize(my_f1_0.to(dev), p=2, dim=0, eps=1e-12, out=None).to(dev)

        # Feature 2: yt - y_t+1|t
        # my_f2_0 = y - torch.squeeze(self.m1y)
        # my_f2_reshape = torch.squeeze(my_f2_0)
        # y_f2_norm = func.normalize(my_f2_reshape, p=2, dim=0, eps=1e-12, out=None)

        # Feature 3: x_t|t - x_t-1|t-1
        m1x_f3_0 = self.m1x_posterior.to(dev) - self.m1x_posterior_previous
        m1x_f3_reshape = torch.squeeze(m1x_f3_0)
        m1x_f3_norm = func.normalize(m1x_f3_reshape, p=2, dim=0, eps=1e-12, out=None)

        # Reshape and Normalize m1x Posterior
        # m1x_post_0 = self.m1x_posterior - self.state_process_posterior_0 # Option 1

        # Featture 4: x_t|t - x_t|t-1
        try:
          m1x_f4_0 = self.m1x_posterior - self.m1x_prior_previous
        except:
          m1x_f4_0 = self.m1x_posterior - self.m1x_prior_previous.unsqueeze(-1)
        m1x_reshape = torch.squeeze(self.m1x_posterior) # Option 3
        m1x_f4_reshape = torch.squeeze(m1x_reshape)
        m1x_f4_norm = func.normalize(m1x_f4_reshape, p=2, dim=0, eps=1e-12, out=None)

        # Normalize y
        # my_0 = y - torch.squeeze(self.obs_process_0) # Option 1
        # my_0 = y - torch.squeeze(self.m1y) # Option 2
        # my_0 = y
        # y_norm = func.normalize(my_0, p=2, dim=0, eps=1e-12, out=None)
        # y_norm = func.normalize(y, p=2, dim=0, eps=1e-12, out=None);

        # Input for counting
        count_norm = func.normalize(torch.tensor([self.i]).float(), dim=0, eps=1e-12, out=None)

        # KGain Net Input
        KGainNet_in = torch.cat([y_f1_norm.to(dev), m1x_f3_norm.to(dev), m1x_f4_norm.to(dev)], dim=1).type(torch.FloatTensor)

        # KGainNet_in = torch.cat([y_f1_norm.to(dev), m1x_f3_norm.to(dev)],dim=0).type(torch.FloatTensor) # m1x_f4_norm.to(dev)], dim=0).type(torch.FloatTensor)
        KGainNet_in = KGainNet_in.to(dev)
        # Kalman Gain Network Step
        KG = self.KGain_step(KGainNet_in.to(dev)).to(dev)
        KG = torch.diag_embed(KG)
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
        # y_obs = torch.unsqueeze(y, 1)
        dy = torch.squeeze(y).to(dev) - self.m1y.to(dev)

        # Compute the 1-st posterior moment
        INOV = torch.matmul(self.KGain.float().to(dev), dy.unsqueeze(-1).float())
        self.m1x_posterior_previous = self.m1x_posterior.to(dev)
        self.m1x_posterior = self.m1x_prior.unsqueeze(-1).to(dev) +INOV.to(dev)

        self.state_process_posterior_0 = self.state_process_prior_0
        self.m1x_prior_previous = self.m1x_prior
        self.y_previous = y

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

        KGainNet_in = KGainNet_in.view(KGainNet_in.size(0), -1)
        # print(KGainNet_in.shape)
        L1_out = self.KG_l1(KGainNet_in)
        La1_out = self.KG_relu1(L1_out)

        ###########
        ### GRU ###
        ###########
        GRU_in = torch.zeros(self.seq_len_input, self.batch_size, self.input_dim)
        GRU_in[0, :, :] = La1_out.to(dev)
        GRU_out, self.hn = self.rnn_GRU(GRU_in.type(torch.FloatTensor).to(dev), self.hn.type(torch.FloatTensor).to(dev))
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
    def forward(self, y):
        yt = torch.squeeze(y)
        yt = self.GFT(yt).to(dev)
        # self.x_out = self.KNet_step(yt)
        return self.IGFT(self.KNet_step(yt.to(dev).to(dev)))

    #########################
    ### Init Hidden State ###
    #########################
    def init_hidden(self):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, self.batch_size, self.hidden_dim).zero_()
        self.hn = hidden.data

    def GFT(self, input):
        return self.BMM_multipy(self.V_t,input.to(dev)).type(torch.FloatTensor)

    def IGFT(self, input):
        # return torch.matmul(self.V, input.to(dev)).type(torch.FloatTensor)
        return self.BMM_multipy(self.V,input.to(dev)).type(torch.FloatTensor)

    def GFT_matrix(self, input):
        return torch.matmul(torch.matmul(self.V_t, input.to(dev)), self.V).type(torch.FloatTensor)
        # return self.BMM_multipy(self.V_t,input.to(dev)).type(torch.FloatTensor)

    def IGFT_matrix(self, input):
        return torch.matmul(torch.matmul(self.V, input.to(dev)), self.V).type(torch.FloatTensor)

    def BMM_multipy(self,a,b):
        if len(b.size())==2:
          b = b.unsqueeze(-1)
        return torch.bmm(a.expand(b.size()[0], -1, -1).type(torch.DoubleTensor).to(dev), b.type(torch.DoubleTensor).to(dev)).squeeze(-1)
