import torch
import torch.nn as nn
import random
import time
import matplotlib.pyplot as plt
from EKFTest import N_T

if torch.cuda.is_available():
    dev = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc.
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print("Running on the GPU")
else:
    dev = torch.device("cpu")
    torch.set_default_tensor_type('torch.FloatTensor')
    print("Running on the CPU")

class Pipeline_EKF:

    def __init__(self, Time, folderName, modelName):
        super().__init__()
        self.Time = Time
        self.folderName = folderName + '/'
        self.modelName = modelName
        self.modelFileName = self.folderName + "model_" + self.modelName
        self.PipelineName = self.folderName + "pipeline_" + self.modelName

    def save(self):
        torch.save(self, self.PipelineName)

    def setssModel(self, ssModel):
        self.ssModel = ssModel

    def setModel(self, model,checkpoint):
        if torch.cuda.is_available():
            self.model = model.to('cuda:0')
        else:
            self.model = model.to('cpu:0')

        # self.model = model.to('cuda:0')
        self.checkpoint = checkpoint
        try:
          self.model.load_state_dict(checkpoint['model_state_dict'])
          print("pre trained 2")
        except:
          print("not pre trained1")

    def setTrainingParams(self, n_Epochs, n_Batch, learningRate, weightDecay):
        self.N_Epochs = n_Epochs  # Number of Training Epochs
        self.N_B = n_Batch  # Number of Samples in Batch
        self.learningRate = learningRate  # Learning Rate
        self.weightDecay = weightDecay  # L2 Weight Regularization - Weight Decay

        # MSE LOSS Function
        self.loss_fn = nn.MSELoss(reduction='mean')

        # Use the optim package to define an Optimizer that will update the weights of
        # the model for us. Here we will use Adam; the optim package contains many other
        # optimization algoriths. The first argument to the Adam constructor tells the
        # optimizer which Tensors it should update.

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)

    def NNTrain(self, n_Examples, train_input, train_target, n_CV, cv_input, cv_target):
        try:
          self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
          print("pre trained 2")
        except:
          print('not loading2')
        # before training the model
        # try:

        # except:
        #   print("Did not load pre-trained model!")

        self.N_E = n_Examples
        self.N_CV = n_CV

        MSE_cv_linear_batch = torch.zeros([self.N_CV])
        self.MSE_cv_linear_epoch = torch.zeros([self.N_Epochs])
        self.MSE_cv_dB_epoch = torch.zeros([self.N_Epochs])

        MSE_train_linear_batch = torch.zeros([self.N_B])
        self.MSE_train_linear_epoch = torch.zeros([self.N_Epochs])
        self.MSE_train_dB_epoch = torch.zeros([self.N_Epochs])

        ##############
        ### Epochs ###
        ##############

        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0

        for ti in range(0, self.N_Epochs):

            ###############################
            ### Training Sequence Batch ###
            ###############################
            self.optimizer.zero_grad()

            # Training Mode
            self.model.train()
            self.model.batch_size = self.N_B

            # Init Hidden State
            self.model.init_hidden()

            # Init Training Batch tensors
            y_training_batch = torch.zeros([self.N_B, self.ssModel.n, self.ssModel.T]).to(dev)
            train_target_batch = torch.zeros([self.N_B, self.ssModel.m, self.ssModel.T]).to(dev)
            x_out_training_batch = torch.zeros([self.N_B, self.ssModel.m, self.ssModel.T]).to(dev)
            Batch_Optimizing_LOSS_sum = 0
            check = True

            # Randomly select N_B training sequences
            n_e = random.sample(range(self.N_E), k=self.N_B)
            ii = 0
            for index in n_e:
                y_training_batch[ii,:,:] = train_input[index]
                train_target_batch[ii,:,:] = train_target[index]
                ii += 1

            self.model.InitSequence(\
                self.ssModel.m1x_0.reshape(1,self.ssModel.m,1).repeat(self.N_B,1,1), self.ssModel.T)

            MSE_trainbatch_linear_LOSS = 0
            # Forward Computation
            for t in range(0, self.ssModel.T):
              x_out_training_batch[:, :, t] = torch.squeeze(self.model(torch.unsqueeze(y_training_batch[:, :, t],2)))
            MSE_trainbatch_linear_LOSS = self.loss_fn(x_out_training_batch[:,:,:], train_target_batch[:,:,:])

            # dB Loss
            self.MSE_train_linear_epoch[ti] = MSE_trainbatch_linear_LOSS.item()
            self.MSE_train_dB_epoch[ti] = 10 * torch.log10(self.MSE_train_linear_epoch[ti])

            ##################
            ### Optimizing ###
            ##################

            # Before the backward pass, use the optimizer object to zero all of the
            # gradients for the variables it will update (which are the learnable
            # weights of the model). This is because by default, gradients are
            # accumulated in buffers( i.e, not overwritten) whenever .backward()
            # is called. Checkout docs of torch.autograd.backward for more details.

            # Backward pass: compute gradient of the loss with respect to model
            # parameters
            MSE_trainbatch_linear_LOSS.backward(retain_graph=True)
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
            # Calling the step function on an Optimizer makes an update to its
            # parameters
            self.optimizer.step()
            # self.scheduler.step(self.MSE_cv_dB_epoch[ti])

            #################################
            ### Validation Sequence Batch ###
            #################################

            # Cross Validation Mode
            self.model.eval()
            self.model.batch_size = self.N_CV
            # Init Hidden State
            self.model.init_hidden()
            with torch.no_grad():

                self.ssModel.T_test = cv_input.size()[-1] # T_test is the maximum length of the CV sequences
                x_out_cv_batch = torch.empty([self.N_CV, self.ssModel.m, self.ssModel.T_test]).to(dev)

                # Init Sequence
                self.model.InitSequence(\
                    self.ssModel.m1x_0.reshape(1,self.ssModel.m,1).repeat(self.N_CV,1,1), self.ssModel.T_test)

                for t in range(0, self.ssModel.T_test):
                    x_out_cv_batch[:, :, t] = torch.squeeze(self.model(torch.unsqueeze(cv_input[:, :, t],2)))

                # Compute CV Loss
                MSE_cvbatch_linear_LOSS = 0
                MSE_cvbatch_linear_LOSS = self.loss_fn(x_out_cv_batch, cv_target)

                # dB Loss
                self.MSE_cv_linear_epoch[ti] = MSE_cvbatch_linear_LOSS.item()
                self.MSE_cv_dB_epoch[ti] = 10 * torch.log10(self.MSE_cv_linear_epoch[ti])

                if (self.MSE_cv_dB_epoch[ti] < self.MSE_cv_dB_opt):
                    self.MSE_cv_dB_opt = self.MSE_cv_dB_epoch[ti]
                    self.MSE_cv_idx_opt = ti

                    # torch.save(self.model, path_results + 'best-model.pt')

            ########################
            ### Training Summary ###
            ########################
            print(ti, "MSE Training :", self.MSE_train_dB_epoch[ti], "[dB]", "MSE Validation :", self.MSE_cv_dB_epoch[ti],
                  "[dB]")

            if (ti > 1):
                d_train = self.MSE_train_dB_epoch[ti] - self.MSE_train_dB_epoch[ti - 1]
                d_cv = self.MSE_cv_dB_epoch[ti] - self.MSE_cv_dB_epoch[ti - 1]
                print("diff MSE Training :", d_train, "[dB]", "diff MSE Validation :", d_cv, "[dB]")
            print("Optimal idx:", self.MSE_cv_idx_opt, "Optimal :", self.MSE_cv_dB_opt, "[dB]")
        self.plot_learning_curve()

    def NNTest(self,n_Test ,test_input, test_target):
        # Load model
        # if load_model:
        #     self.model = torch.load(load_model_path, map_location=self.device)
        # else:
        #     self.model = torch.load(path_results+'best-model.pt', map_location=self.device)

        self.N_T = n_Test
        # SysModel.T_test = test_input.size()[-1]
        self.MSE_test_linear_arr = torch.zeros([self.N_T])
        SysModel = self.ssModel
        x_out_test = torch.zeros([self.N_T, SysModel.m,SysModel.T_test]).to(dev)

        # if MaskOnState:
        #     mask = torch.tensor([True,False,False])
        #     if SysModel.m == 2:
        #         mask = torch.tensor([True,False])

        # MSE LOSS Function
        loss_fn = nn.MSELoss(reduction='mean')

        # Test mode
        self.model.eval()
        self.model.batch_size = self.N_T
        # Init Hidden State
        self.model.init_hidden()
        torch.no_grad()

        start = time.time()
        self.model.InitSequence(SysModel.m1x_0.reshape(1,SysModel.m,1).repeat(self.N_T,1,1), SysModel.T_test)

        for t in range(0, SysModel.T_test):
            x_out_test[:,:, t] = torch.squeeze(self.model(torch.unsqueeze(test_input[:,:, t],2)))

        end = time.time()
        t = end - start
        print("se",start,end,end-start)
        MSE_finalize = 0
        # MSE loss
        # print("Estimation: ",x_out_test[5,:,SysModel.T_test-1])
        # print("Target: ",test_target[5,:,SysModel.T_test-1])
        MSE_test_per_iter = torch.zeros(test_target.shape[2])
        for j in range(self.N_T):# cannot use batch due to different length and std computation
          self.MSE_test_linear_arr[j] = loss_fn(x_out_test[j,:,:], test_target[j,:,:]).item()
          for t in range(0, SysModel.T_test):
            MSE_test_per_iter[t] = loss_fn(x_out_test[j,:,t], test_target[j, :, t])
          MSE_finalize += MSE_test_per_iter

        # Average
        self.MSE_test_linear_avg = torch.mean(self.MSE_test_linear_arr)
        self.MSE_test_dB_avg = 10 * torch.log10(self.MSE_test_linear_avg)
        MSE_finalize = MSE_finalize / N_T # average
        MSE_finalize_dB = 10 * torch.log10(MSE_finalize)
        # print("MSE finalize dB",MSE_finalize_dB)
        # Standard deviation
        self.MSE_test_linear_std = torch.std(self.MSE_test_linear_arr, unbiased=True)

        # Confidence interval
        self.test_std_dB = 10 * torch.log10(self.MSE_test_linear_std + self.MSE_test_linear_avg) - self.MSE_test_dB_avg

        # Print MSE and std
        str = self.modelName + "-" + "MSE Test:"
        print(str, self.MSE_test_dB_avg, "[dB]")
        str = self.modelName + "-" + "STD Test:"
        print(str, self.test_std_dB, "[dB]")
        # Print Run Time
        print("Inference Time:", t)

        return [self.MSE_test_linear_arr, self.MSE_test_linear_avg, self.MSE_test_dB_avg, x_out_test, t]



    def plot_learning_curve(self):
        iters_sub = [i for i in range(self.N_Epochs)]
        plt.title("Learning Curve: Accuracy per Iteration")
        plt.plot(iters_sub, self.MSE_train_dB_epoch.cpu(), label="Train")
        plt.plot(iters_sub, self.MSE_cv_dB_epoch.cpu(), label="Validation")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy - MSE")
        plt.legend(loc='best')
        plt.show()