import torch
import torch.nn as nn
import random
import torch.nn.functional as func
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from torch import autograd
from EKF import ExtendedKalmanFilter
global N_T
def EKFTest(SysModel, test_input, test_target, equation=13, model='regular',  modelKnowledge='full', allStates=True):
    N_T = test_target.size()[0]

    # LOSS
    loss_fn = nn.MSELoss(reduction='mean')

    # MSE [Linear]
    MSE_EKF_linear_arr = torch.zeros(N_T)
    SNR = torch.zeros(N_T)
    EKF = ExtendedKalmanFilter(SysModel, equation, model, modelKnowledge)
    EKF.InitSequence(SysModel.m1x_0, SysModel.m2x_0)
    MSE_finalize = torch.zeros(test_target.shape[2])
    KG_array = torch.zeros_like(EKF.KG_array)
    EKF_out = torch.zeros([N_T, SysModel.m, SysModel.T_test])
    start = time.time()
    MSE_test_per_iter = torch.zeros(test_target.shape[2])
    stoper = True
    for j in range(0, N_T):
        # print(test_input.shape)
        EKF.GenerateSequence(test_input[j, :, :], EKF.T_test)
        for t in range(0, SysModel.T):
            MSE_test_per_iter[t] = loss_fn(EKF.x[:, t], test_target[j, :, t])

        SNR[j] = 1  # torch.sqrt(sum(abs(test_target[:, j+1] - EKF.x[:, j])))
        MSE_EKF_linear_arr[j] = loss_fn(EKF.x, test_target[j, :, :]).item()
        KG_array = torch.add(EKF.KG_array, KG_array)
        EKF_out[j, :, :] = EKF.x
        MSE_finalize += MSE_test_per_iter

    end = time.time()
    t = end-start

    # Average KG_array over Test Examples
    KG_array /= N_T

    # print('SNR', SNR)
    SNR_avg = torch.mean(SNR)/(SysModel.r**2)
    SNR_dB_avg = 10 * torch.log10(SNR_avg)

    # print('MSE', MSE_EKF_linear_arr)
    MSE_EKF_linear_avg = torch.mean(MSE_EKF_linear_arr)
    MSE_EKF_dB_avg = 10 * torch.log10(MSE_EKF_linear_avg)

    # Standard deviation
    MSE_EKF_dB_std = torch.std(MSE_EKF_linear_arr, unbiased=True)
    MSE_EKF_dB_std = 10 * torch.log10(MSE_EKF_dB_std)
    print(model)
    print("EKF - MSE LOSS:", MSE_EKF_dB_avg, "[dB]")

    # Print Run Time
    print("Inference Time:", t)

    MSE_finalize = MSE_finalize / N_T # average
    MSE_finalize_dB = 10 * torch.log10(MSE_finalize)
    if torch.isnan(MSE_finalize).any() == True:
      print("the nan is in the finalize dB!!")

    return [MSE_finalize_dB, MSE_EKF_linear_arr, MSE_EKF_linear_avg, MSE_EKF_dB_avg, KG_array, EKF_out, SNR_dB_avg]