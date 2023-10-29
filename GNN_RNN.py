import torch.optim as optim
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import dgl
from torch.utils.data import Dataset
import math
import time

def generate_laplacian_matrix(n):
    if n <= 0:
        raise ValueError("The number of nodes (n) must be a positive integer.")

    # Create a symmetric adjacency matrix with random weights
    adjacency_matrix = torch.rand(n, n)
    adjacency_matrix = (adjacency_matrix + adjacency_matrix.t()) / 2.0  # Ensure symmetry

    # Compute the degree matrix
    degree_matrix = torch.diag(adjacency_matrix.sum(dim=1))

    # Compute the Laplacian matrix
    laplacian_matrix = degree_matrix - adjacency_matrix

    return laplacian_matrix

# Example: Generate a Laplacian matrix for a graph with 4 nodes
n = 300
laplacian_matrix = generate_laplacian_matrix(n)

def laplacian_to_adjacency(laplacian):
    n = laplacian.size(0)  # Number of nodes in the graph
    adjacency = torch.zeros((n, n), dtype=laplacian.dtype, device=laplacian.device)

    for i in range(n):
        for j in range(n):
            if i != j:
                adjacency[i][j] = -laplacian[i][j]
            else:
                adjacency[i][j] = laplacian[i].sum()  # Degree of node i

    return adjacency

# Example Laplacian matrix for a weighted graph as a PyTorch tensor
# L14 = np.array([
#     [19.4980702055144, -15.2630865231796, 0, 0, -4.23498368233483, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [-15.2630865231796, 30.3547153987791, -4.78186315175772, -5.11583832587208, -5.19392739796971, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, -4.78186315175772, 9.85068012935164, -5.06881697759392, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, -5.11583832587208, -5.06881697759392, 38.3431317384716, -21.5785539816916, 0, -4.78194338179036, 0, -1.79797907152361, 0, 0, 0, 0, 0],
#     [-4.23498368233483, -5.19392739796971, 0, -21.5785539816916, 34.9754041144523, -3.96793905245615, 0, 0, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, -3.96793905245615, 17.3407328099191, 0, 0, 0, 0, -4.09407434424044, -3.17596396502940, -6.10275544819312, 0],
#     [0, 0, 0, -4.78194338179036, 0, 0, 19.5490059482647, -5.67697984672154, -9.09008271975275, 0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0, 0, -5.67697984672154, 5.67697984672154, 0, 0, 0, 0, 0, 0],
#     [0, 0, 0, -1.79797907152361, 0, 0, -9.09008271975275, 0, 24.2825063752679, -10.3653941270609, 0, 0, 0, -3.02905045693060],
#     [0, 0, 0, 0, 0, 0, 0, 0, -10.3653941270609, 14.7683378765214, -4.40294374946052, 0, 0, 0],
#     [0, 0, 0, 0, 0, -4.09407434424044, 0, 0, 0, -4.40294374946052, 8.49701809370096, 0, 0, 0],
#     [0, 0, 0, 0, 0, -3.17596396502940, 0, 0, 0, 0, 0, 5.42793859120161, -2.25197462617221, 0],
#     [0, 0, 0, 0, 0, -6.10275544819312, 0, 0, 0, 0, 0, -2.25197462617221, 10.6696935494707, -2.31496347510535],
#     [0, 0, 0, 0, 0, 0, 0, 0, -3.02905045693060, 0, 0, 0, -2.31496347510535, 5.34401393203596]
# ])

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

L30 = path_test_target = f'L_30.pt' # from Non linear n = 30
L30 = torch.load(L30)

# laplacian_matrix = torch.tensor([[2.7, -1.0, 0.0],
#                                   [-1.0, 3.0, -1.0],
#                                   [0.0, -1.0, 2.0]], dtype=torch.float32)

# Generate the adjacency matrix from the Laplacian matrix for a weighted graph
adjacency_matrix = laplacian_to_adjacency(laplacian_matrix)

print("Laplacian Matrix:")
print(laplacian_matrix.shape)
print("\nAdjacency Matrix (for a weighted graph):")
# print(adjacency_matrix)




def create_edges(adj):
    edge_coords = torch.nonzero(adj, as_tuple=True)
    edge_list = [(int(src), int(dst)) for src, dst in zip(edge_coords[0], edge_coords[1])]
    return edge_list

def evaluation(model, graph,dataloader,dataset_name,r2):
    criterion = nn.MSELoss()
    model.eval()
    loss_list=[]
    with torch.no_grad():
        for target_val, obs_val in dataloader:
            # sta = time.time()
            output_val = model(graph, obs_val.squeeze(0))
            # end = time.time()
            # print(end-sta)
            loss_val = criterion(output_val, target_val.squeeze(0))
            loss_list.append(loss_val)
    loss = torch.mean(torch.stack(loss_list)).item()
    torch.save(model.state_dict(), f'./models/Non Linear H x3/gnn_model_vis1_{r2}.pth')
    return loss

def load_data(r2,dataset):
  # /content/drive/MyDrive/Project/data/pypower14f
    path_test_target = f'./data/{dataset}/test_target_r2_tensor({r2}).pt'
    path_test_obs = f'./data/{dataset}/test_observation_r2_tensor({r2}).pt'
    test_target = torch.load(path_test_target)
    test_obs = torch.load(path_test_obs)

    # path_train_target = f'./data/{dataset}/train_target_r2_tensor({r2}).pt'
    # path_train_obs = f'./data/{dataset}/train_observation_r2_tensor({r2}).pt'
    # train_target = torch.load(path_train_target)
    # train_obs = torch.load(path_train_obs)

    # path_val_target = f'./data/{dataset}/valid_target_r2_tensor({r2}).pt'
    # path_val_obs = f'./data/{dataset}/valid_observation_r2_tensor({r2}).pt'
    # val_target = torch.load(path_val_target)
    # val_obs = torch.load(path_val_obs)

    return test_target, test_obs,test_target, test_obs,test_target, test_obs

    # return test_target, test_obs, train_target, train_obs, val_target, val_obs

adj_pyp = torch.tensor([[-3.1974e-14,  1.5263e+01, -0.0000e+00, -0.0000e+00,  4.2350e+00,
         -0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00,
         -0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00],
        [ 1.5263e+01, -1.1546e-14,  4.7819e+00,  5.1158e+00,  5.1939e+00,
         -0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00,
         -0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00],
        [-0.0000e+00,  4.7819e+00,  0.0000e+00,  5.0688e+00, -0.0000e+00,
         -0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00,
         -0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00],
        [-0.0000e+00,  5.1158e+00,  5.0688e+00,  2.8422e-14,  2.1579e+01,
         -0.0000e+00,  4.7819e+00, -0.0000e+00,  1.7980e+00, -0.0000e+00,
         -0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00],
        [ 4.2350e+00,  5.1939e+00, -0.0000e+00,  2.1579e+01,  1.4211e-14,
          3.9679e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00,
         -0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00],
        [-0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00,  3.9679e+00,
         -1.0658e-14, -0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00,
          4.0941e+00,  3.1760e+00,  6.1028e+00, -0.0000e+00],
        [-0.0000e+00, -0.0000e+00, -0.0000e+00,  4.7819e+00, -0.0000e+00,
         -0.0000e+00,  4.9738e-14,  5.6770e+00,  9.0901e+00, -0.0000e+00,
         -0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00],
        [-0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00,
         -0.0000e+00,  5.6770e+00,  0.0000e+00, -0.0000e+00, -0.0000e+00,
         -0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00],
        [-0.0000e+00, -0.0000e+00, -0.0000e+00,  1.7980e+00, -0.0000e+00,
         -0.0000e+00,  9.0901e+00, -0.0000e+00,  4.0856e-14,  1.0365e+01,
         -0.0000e+00, -0.0000e+00, -0.0000e+00,  3.0291e+00],
        [-0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00,
         -0.0000e+00, -0.0000e+00, -0.0000e+00,  1.0365e+01, -2.0428e-14,
          4.4029e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00],
        [-0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00,
          4.0941e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00,  4.4029e+00,
          1.7764e-15, -0.0000e+00, -0.0000e+00, -0.0000e+00],
        [-0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00,
          3.1760e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00,
         -0.0000e+00,  0.0000e+00,  2.2520e+00, -0.0000e+00],
        [-0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00,
          6.1028e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00,
         -0.0000e+00,  2.2520e+00,  2.1316e-14,  2.3150e+00],
        [-0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00, -0.0000e+00,
         -0.0000e+00, -0.0000e+00, -0.0000e+00,  3.0291e+00, -0.0000e+00,
         -0.0000e+00, -0.0000e+00,  2.3150e+00,  1.0214e-14]])

# adj_nonlinear = torch.tensor([[0., 0., -0., 1., 1., 1., 1., -0., -0., 1.],
#         [0., 0., -0., 1., -0., 1., 1., 1., -0., 1.],
#         [-0., -0., 0., 1., 1., 1., -0., 1., 1., 1.],
#         [1., 1., 1., 0., 1., -0., 1., -0., -0., 1.],
#         [1., -0., 1., 1., 0., -0., 1., 1., 1., -0.],
#         [1., 1., 1., -0., -0., 0., -0., 1., 1., 1.],
#         [1., 1., -0., 1., 1., -0., 0., 1., 1., -0.],
#         [-0., 1., 1., -0., 1., 1., 1., 0., 1., -0.],
#         [-0., -0., 1., -0., 1., 1., 1., 1., 0., 1.],
#         [1., 1., 1., 1., -0., 1., -0., -0., 1., 0.]])

adj_nonlinear = torch.tensor([[0., 1., -0., -0., -0., -0., 1., -0., -0., 1.],
        [1., 0., -0., -0., -0., -0., -0., -0., 1., 1.],
        [-0., -0., 0., 1., -0., 1., 1., -0., -0., -0.],
        [-0., -0., 1., 0., 1., -0., -0., -0., -0., 1.],
        [-0., -0., -0., 1., 0., 1., -0., 1., -0., -0.],
        [-0., -0., 1., -0., 1., 0., -0., -0., 1., -0.],
        [1., -0., 1., -0., -0., -0., 0., 1., -0., -0.],
        [-0., -0., -0., -0., 1., -0., 1., 0., 1., -0.],
        [-0., 1., -0., -0., -0., 1., -0., 1., 0., -0.],
        [1., 1., -0., 1., -0., -0., -0., -0., -0., 0.]])


def invert_tensors_average(n, k):
    total_time = 0.0

    for _ in range(k):
        # Create a random square matrix as an example
        tensor_to_invert = torch.rand(n, n)

        # Measure the time taken to invert the tensor
        start_time = time.time()
        inverted_tensor = torch.inverse(tensor_to_invert)
        end_time = time.time()

        # Calculate the elapsed time
        elapsed_time = end_time - start_time
        total_time += elapsed_time

    average_time = total_time / k
    return average_time

# Define the size of the tensor (n x n) and the number of tensors (k)
n = 1000
k = 100  # You can adjust this based on the number of tensors you want to invert

average_inversion_time = invert_tensors_average(n, k)
print(f"Average time taken to invert {k} {n}x{n} tensors: {average_inversion_time:.4f} seconds")


def exponential(x, lambda_exp):
  """Generates exponential noise with parameter lambda_exp."""
  u = torch.rand(x.shape)
  return -1.0 * torch.log(1-u) * lambda_exp

# Add exponential noise to the tensor `test_Y`
exponential(torch.tensor([500,10]), 0.009)



################## Define Models #########################
# GNN
class GNNModel(nn.Module):
    def __init__(self, in_feats, hidden_feats):
        super(GNNModel, self).__init__()
        self.gconv1 = dgl.nn.GraphConv(in_feats, hidden_feats, activation=nn.ReLU())
        self.gconv2 = dgl.nn.GraphConv(hidden_feats, hidden_feats, activation=nn.ReLU())
    def forward(self, graph, node_features):
        gnn_out = self.gconv1(graph, node_features)
        gnn_out = self.gconv2(graph, gnn_out)
        return gnn_out

# RNN
class RNNModel(nn.Module):
    def __init__(self, input_rnn_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.rnn1  = nn.LSTM(input_size=input_rnn_size, hidden_size=hidden_size, num_layers=2, batch_first=True)
        self.rnn2 = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    def forward(self, input_seq):
        rnn_output, _ = self.rnn1(input_seq)
        rnn_output, _ = self.rnn2(rnn_output)
        rnn_output = self.fc1(rnn_output)
        output = self.fc2(rnn_output)
        return output

# Overall model
class TrajectoryModel(nn.Module):
    def __init__(self, gnn_model, rnn_model):
        super(TrajectoryModel, self).__init__()
        self.gnn_model = gnn_model
        self.rnn_model = rnn_model
    def forward(self, graph, batch):
        gnn_output = self.gnn_model(graph, batch)
        preds_list = []
        for node_idx in range(batch.shape[0]):
            rnn_input = gnn_output[node_idx,:,:]
            node_pred = self.rnn_model(rnn_input)
            preds_list.append(node_pred.unsqueeze(0))
        out = torch.concat(preds_list, dim=0)
        return out

############################# define dataset ####################
class GraphDataset(Dataset):
    def __init__(self, target, obs):
        self.num_samples = len(target)
        self.data = []
        for k in range(len(target)):
            target_k = target[k].unsqueeze(2).float()
            obs_k = obs[k].unsqueeze(2).float()
            self.data.append((target_k,obs_k))
    def __len__(self):
        return self.num_samples
    def __getitem__(self, idx):
        return self.data[idx]





# Hyper parameters
num_samples = 100
batch_size = 1
num_nodes = 14  # Number of nodes in the fixed graph
num_timesteps = 100
input_rnn_size = 3*num_nodes
hidden_size = num_nodes+4
criterion = nn.MSELoss()
LR = 0.001

# Load the graph signals data
r2='0.5000' # 0.005, 0.01, 0.05, 0.1
dataset_name="case57" #"50test" #"Hx3-visual" #'pypower14-x0.10.001xA1x' #'Non Linear H x3/n=10', 'pypower14f'
adj_binary = adj_pyp  #adjacency_matrix
load_flag = False
train_flag = True
print(f'*********Dataset: {dataset_name} with r2={r2}**********')

test_target, test_obs, train_target, train_obs, val_target, val_obs = load_data(r2,dataset_name)
# create loader
dataset_train = GraphDataset(train_target, train_obs)
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
dataset_val = GraphDataset(val_target, val_obs)
dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)
dataset_test = GraphDataset(test_target, test_obs)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

# Generate graph structure
if dataset_name=='Hx3-visual':
    edges = create_edges(adjacency_matrix)
else:
    edges = [(i, i + 1) for i in range(train_obs.shape[1]-1)]
    edges.append((13,0))
graph = dgl.graph(edges)

# Initialize the GNN and RNN models
gnn_model = GNNModel(1, input_rnn_size)
rnn_model = RNNModel(input_rnn_size, hidden_size, 1)
model = TrajectoryModel(gnn_model, rnn_model)
optimizer = optim.Adam(model.parameters(), lr=LR)
if load_flag:
    model.load_state_dict(torch.load(f'./models/Non Linear H x3/gnn_model_vis12_{r2}.pth'))
    print("loading model")
else:
  print("training a new model")

if train_flag:
    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        k=0
        for target_batch, obs_batch in dataloader_train:
            if (k%2==0):
              print(k)
            if k >= 20:
              break
            optimizer.zero_grad()
            outputs = model(graph, obs_batch.squeeze(0))
            loss = criterion(outputs, target_batch.squeeze(0))
            loss.backward()
            optimizer.step()
            k+=1
        loss_val = evaluation(model,graph, dataloader_val,dataset_name,r2)
        print(f"Epoch [{epoch + 1}/{num_epochs}] Loss train: {round(10*math.log10(loss.item()),6)} dB, Loss val: {round(10*math.log10(loss_val),6)} dB")
start = time.time()
loss_test = evaluation(model,graph, dataloader_test,dataset_name,r2)
end = time.time()
print("Time inference: ", end-start)
print(f"Loss test: {round(10*math.log10(loss_test),6)} dB")

non_linear=[-18.6,-24.36,-29, -32.26]
noise_r2=[1.,0.1000, 0.0100,0.0010]


output = model(graph,(test_obs[19,:,:].unsqueeze(2).float()))
output.squeeze()
print(output[5,:])




