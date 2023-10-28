import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pypower.api import case118, makeYbus, case14, case30
from pypower.idx_brch import F_BUS, T_BUS

if torch.cuda.is_available():
    dev = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print("Running on the GPU")
else:
    dev = torch.device("cpu")
    torch.set_default_tensor_type('torch.FloatTensor')
    print("Running on the CPU")

from datetime import datetime
from GSP_Extended_Sysmodel import SystemModel
from EKFTest import EKFTest
from Power_Grid_Simulation_Parameters import pypower_m, pypower_n, pypower_f, pypower_h, pypower_L, pypower_V, pypower_V_t, pypower_f_EKF, pypower_h_EKF
from Pipline_EKF import Pipeline_EKF
from Ex_Kalmannet import ExtendedKalmanNetNN
from GSP_EXtended_Kalmannet import GSPExtendedKalmanNetNN
import torch.nn.init as init

def init_weights(module):
    if isinstance(module, nn.Linear):
        init.xavier_uniform_(module.weight)
        module.bias.data.fill_(0.0)



T = 100
T_valid = T
T_test = T
nl_T = T
nl_T_test = nl_T
nl_m = pypower_m  # In case of nonlinear need to change to 10 or nl_m parameter from Non_Linear_Parameters
nl_n = pypower_n  # In case of nonlinear need to change to 10 or nl_n parameter from Non_Linear_Parameters

PFandUKF_test = False
if torch.cuda.is_available() and not PFandUKF_test:
    dev = torch.device("cuda:0")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print("Running on the GPU")
else:
    dev = torch.device("cpu")
    torch.set_default_tensor_type('torch.FloatTensor')
    print("Running on the CPU")
print("Pipeline Start")

################
### Get Time ###
################

today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H:%M:%S")
strTime = strToday + "_" + strNow
print("Current Time =", strTime)

N_E = 4000  # train
N_CV = 100  # validation
N_T = 100   # test

# All together
r2 = torch.tensor([0.005, 0.01, 0.05, 0.1,0.5])  # for Exponential noise case57 data
# r2 = torch.tensor([0.001, 0.01, 0.1, 1, 10])  # for gaussian noise
r = torch.sqrt(r2)
vdB = -20
# ratio v=q2/r2
v = 10 ** (vdB / 10)
q2 = torch.mul(v, r2)
q = torch.sqrt(q2)
EKF_result = torch.empty([len(r2), nl_m, T_test])
epsilon = torch.eye(nl_n)*(1E-11)

nl_GSP_KNet_MSElist = []
nl_KNet_MSElist = []
nl_MSElist = []
nl_MSElist_diag = []
SNR_RW_13 = []
SNR_RW_20_diag = []
Naive_MSE = []

###### Decide which test to run #######
kalmanNet = False
GSP_KalmanNet = False
GSP_EKF = False
EKF = True

for index in range(len(r2)):

    print("1/r2 [dB]: ", 10 * torch.log10(1 / r[index] ** 2), r2[index])
    print("1/q2 [dB]: ", 10 * torch.log10(1 / q[index] ** 2))

    target = torch.load('./data/case57/test_target_r2_'+ str(r2[index]) + '.pt').to(dev)
    observation = torch.load('./data/case57/test_observation_r2_' + str(r2[index]) + '.pt').to(dev)
    print(target.size())
    print(observation.size())
    target = target[:N_T,:,:T_test].float()
    observation = observation[:N_T,:,:T_test].float()
    nl_train_input = torch.load('./data/case57/train_observation_r2_' + str(r2[index]) + '.pt').to(dev)
    nl_cv_input = torch.load('./data/case57/valid_observation_r2_' + str(r2[index]) + '.pt').to(dev)
    nl_test_input = observation
    nl_train_target = torch.load('./data/case57/train_target_r2_' + str(r2[index]) + '.pt').to(dev)
    nl_cv_target = torch.load('./data/case57/valid_target_r2_' + str(r2[index]) + '.pt').to(dev)
    nl_test_target = target
    nl_train_input = nl_train_input[:,:,:T].float()
    nl_cv_input = nl_cv_input[:N_CV,:,:T_valid].float()
    nl_test_input = nl_test_input[:,:,:T].float()
    nl_train_target = nl_train_target[:,:,:T].float()
    nl_cv_target = nl_cv_target[:N_CV,:,:T_valid].float()
    nl_test_target = nl_test_target[:,:,:T].float()

    ### Pypower
    m1x_0 = target[0, :, 0].to(dev)
    m2x_0 = 0 * 0 * torch.zeros(nl_m, nl_m).to(dev)

    ######## KalmanNet ########
    if kalmanNet:
        sys_model = SystemModel(pypower_f, q[index], pypower_h, r[index], nl_T, nl_T_test, nl_m, nl_n, pypower_L, pypower_V, pypower_V_t)
        sys_model.InitSequence(m1x_0, m2x_0)
        print("KNet EKF with full model info")
        modelFolder = 'KNet' + '/'
        KNet_Pipeline = Pipeline_EKF(strTime, "KNet", "KalmanNet")
        KNet_Pipeline.setssModel(sys_model)
        KNet_model = ExtendedKalmanNetNN()
        try:
          checkpoint = torch.load('/content/drive/MyDrive/Project/models/case57/KNet_weights_f'+str(r2[index-1])+'.pth')
          epochs = 80
        except:
          checkpoint = 0
          epochs = 120
          print("Training from zero")
          # KNet_model.apply(init_weights)
        KNet_model.Build(sys_model)
        KNet_model.apply(init_weights)
        KNet_Pipeline.setModel(KNet_model, checkpoint)
        KNet_Pipeline.setTrainingParams(n_Epochs=epochs, n_Batch=100 , learningRate=1e-4, weightDecay=0)
        if epochs != 2:
          KNet_Pipeline.NNTrain(N_E, nl_train_input, nl_train_target, N_CV, nl_cv_input, nl_cv_target)
          print("Saving model ----->")
          torch.save({
                  'model_state_dict': KNet_model.state_dict(),
                  'optimizer_state_dict': KNet_Pipeline.optimizer.state_dict()
                  }, '/content/drive/MyDrive/Project/models/case57/KNet_weights_f'+str(r2[index-1])+'.pth')
        [KNet_MSE_test_linear_arr, KNet_MSE_test_linear_avg, KNet_MSE_test_dB_avg, KNet_test, timming] = KNet_Pipeline.NNTest(N_T,
                                                                                                                nl_test_input,
                                                                                                                  nl_test_target)
        nl_KNet_MSElist.append(KNet_MSE_test_dB_avg)

    ######## GSP-KalmanNet ########
    if GSP_KalmanNet:
        sys_model = SystemModel(pypower_f, q[index], pypower_h, r[index], nl_T, nl_T_test, nl_m, nl_n, pypower_L, pypower_V, pypower_V_t)
        sys_model.InitSequence(m1x_0.to(dev), m2x_0.to(dev))
        print("GSP - KNet EKF with full model info")
        modelFolder = 'KNet' + '/'
        gspKNet_Pipeline = Pipeline_EKF(strTime, "GSP-KalmanNet", "GSP-KalmanNet")
        gspKNet_Pipeline.setssModel(sys_model)
        gspKNet_model = GSPExtendedKalmanNetNN()
        try:
          checkpoint = torch.load('/content/drive/MyDrive/Project/models/case57/gspKNet_weights_f'+str(r2[index-1])+'.pth')
          epochs = 2
          if index > 1:
            epochs = 50
          print("from checkpoint")
        except:
          checkpoint = 0
          epochs = 200
          gspKNet_model.apply(init_weights)
          print("Training from zero")
        gspKNet_model.Build(sys_model)
        gspKNet_Pipeline.setModel(gspKNet_model,checkpoint)
        gspKNet_Pipeline.setTrainingParams(n_Epochs=epochs, n_Batch=100, learningRate=1e-4, weightDecay=0)
        if epochs != 2:
          gspKNet_Pipeline.NNTrain(N_E, nl_train_input, nl_train_target, N_CV, nl_cv_input, nl_cv_target)
          print("Saving model ----->")
          torch.save({
                  'model_state_dict': gspKNet_model.state_dict(),
                  'optimizer_state_dict': gspKNet_Pipeline.optimizer.state_dict()
                  }, '/content/drive/MyDrive/Project/models/case57/gspKNet_weights_f'+str(r2[index])+'.pth')

        [KNet_MSE_test_linear_arr, KNet_MSE_test_linear_avg, gspKNet_MSE_test_dB_avg, gspKNet_test, timming] = gspKNet_Pipeline.NNTest(N_T,
                                                                                                                     nl_test_input,
                                                                                                                     nl_test_target)
        nl_GSP_KNet_MSElist.append(gspKNet_MSE_test_dB_avg)


    ######### gsp-EKF ###########
    if GSP_EKF:
        mean = torch.mean(nl_test_target, dim=2)
        # subtract the mean from the data
        centered_data = nl_test_target - mean.unsqueeze(2)
        # compute the covariance matrix of the data
        covariance_matrix = torch.matmul(centered_data, centered_data.transpose(1, 2)) / (centered_data.size(2) - 1)
        avg_covariance_matrix = torch.mean(covariance_matrix, dim=0)
        m2x_0 = covariance_matrix[0,:,:]
        epsilon = torch.eye(pypower_m)*(1E-13)
        equation = 13
        if torch.isnan(m2x_0).any():
            print("############## Variable F contains NaN values.")

        sys_model = SystemModel(pypower_f_EKF, q[index].to(dev), pypower_h_EKF, r[index].to(dev)*26, nl_T, nl_T_test, nl_m, nl_n, pypower_L.to(dev), pypower_V.to(dev), pypower_V_t.to(dev))
        sys_model.InitSequence(m1x_0, m2x_0)

        # Diagonal-KG gsp-EKF
        print("GSP-EKF")
        equation = 20
        [MSE_finalize20, nl20_MSE_EKF_linear_arr, nl20_MSE_EKF_linear_avg, nl20_MSE_EKF_dB_avg, nl20_EKF_KG_array, nl20_EKF_out,
        nl20_SNR] = EKFTest(sys_model, nl_test_input, nl_test_target, equation, 'Pypower')
        if torch.isnan(nl20_EKF_KG_array).any() or torch.isnan(nl20_MSE_EKF_linear_arr).any():
            print("Output KG or MSE array contains NaN values!!!!!!!")

        nl_MSElist_diag.append(nl20_MSE_EKF_dB_avg)

    ######## EKF ###########
    if EKF:
        mean = torch.mean(nl_test_target, dim=2)
        centered_data = nl_test_target - mean.unsqueeze(2)
        covariance_matrix = torch.matmul(centered_data, centered_data.transpose(1, 2)) / (centered_data.size(2) - 1)
        avg_covariance_matrix = torch.mean(covariance_matrix, dim=0)
        m2x_0 = covariance_matrix[0,:,:]
        epsilon = torch.eye(nl_m)*(1E-13)
        equation = 13
        I = torch.eye(nl_m)
        if torch.isnan(m2x_0).any():
            print("############## Variable F contains NaN values.")

        sys_model = SystemModel(pypower_f_EKF, q[index].to(dev), pypower_h_EKF, r[index].to(dev)*26, nl_T, nl_T_test, nl_m, nl_n, I.to(dev), I.to(dev), I.to(dev))
        sys_model.InitSequence(m1x_0, m2x_0)

        [MSE_finalize13, nl13_MSE_EKF_linear_arr, nl13_MSE_EKF_linear_avg, nl13_MSE_EKF_dB_avg, nl13_EKF_KG_array, nl13_EKF_out,
        nl13_SNR] = EKFTest(sys_model, nl_test_input, nl_test_target, equation, 'Pypower')
        if torch.isnan(nl13_EKF_KG_array).any() or torch.isnan(nl13_MSE_EKF_linear_arr).any():
            print("Output KG or MSE array contains NaN values!!!!!!!")

    # # Average
    # MSE_test_linear_avg = torch.mean(MSE_test_linear_arr)
    # MSE_test_dB_avg = 10 * torch.log10(MSE_test_linear_avg)
    # mse = torch.mean((target - estimation)**2)
    # print("Naive MSE ", MSE_test_dB_avg, " [dB]")
    # Naive_MSE.append(MSE_test_dB_avg)
    # # plt.figure()

    # plt.plot(range(1,T), MSE_test_linear_arr[1:].cpu().detach().numpy(), label='Knet')
    # # plt.plot(range(2, T+1), KF_MSE_finalize_dB[1:].cpu().detach().numpy(), label='KF')
    # # plt.plot(range(2, T+1), MSE_finalize_gspKnet[1:].cpu().detach().numpy(), label='Knet GSP')
    # # plt.plot(range(2, T+1), MSE_finalize13[1:].cpu().detach().numpy(), label='GSP-EKF-full')
    # # plt.plot(range(2, T+1), MSE_finalize20[1:].cpu().detach().numpy(), label='GSP-EKF-diag')
    # plt.xlabel('Iteration')
    # plt.ylabel('MSE dB')
    # plt.title(' MSE vs Iteration')
    # plt.legend()
    # plt.grid()
    # plt.show()