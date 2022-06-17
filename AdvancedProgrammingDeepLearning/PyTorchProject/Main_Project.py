
################################### Libraries ##################################################
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import  torch_geometric.data as data
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures , AddSelfLoops
#################################################################################################


############################################ Data Collection ###################################
dataset = Planetoid(root='data/Planetoid', name='Cora', transform=None ) #NormalizeFeatures
print()
print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')
data = dataset[0]
#################################################################################################




##################################### Model and Data Preparation ##########################################
class MLP(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(12345)
        self.lin1 = Linear(dataset.num_features, hidden_channels)
        self.lin2 = Linear(hidden_channels, dataset.num_classes)
    def forward(self, x):
        x = self.lin1(x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(1234567)
        self.conv1 = GCNConv(dataset.num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# How do we choose the dimension of the hidden layers ?
# There is no way of knowing in advance, the main idea is that we should reduce the dimensions  and iterate
# In literature they propose 16 as the best number of hidden layers
n_hidden_dim = 16
#################################################################################################




################################## MLP  ########################################################
# Use MLP model as model
model = MLP(hidden_channels=n_hidden_dim)
# Define loss criterion.
criterion = torch.nn.CrossEntropyLoss()
# Define Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)  # Define optimizer.
print(model)
def train():
      # Train option
      model.train()
      # Clear gradients.
      optimizer.zero_grad()
      # Perform a single forward pass for the iteration i
      out = model(data.x)
      # Compute the loss solely based on the training nodes.
      loss = criterion(out[data.train_mask], data.y[data.train_mask])
      # Derive gradients.
      loss.backward()
      # Update parameters based on gradients.
      optimizer.step()
      return loss

def test():
      # Model is in evaluation mode
      model.eval()
      # Use model to make prediction
      out = model(data.x)
      # Use the class with highest probability since the output is a vector of softmax
      pred = out.argmax(dim=1)
      # Compute accuracy as the counter of  prediction vs ground-truth labels.
      test_correct = pred[data.test_mask] == data.y[data.test_mask]
      # Derive ratio of correct predictions. to get Accuracy
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
      return test_acc

for epoch in range(1, 201):
    loss = train()
    #print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
test_acc_mlp = test()
print(f'Test Accuracy: {test_acc_mlp:.4f}')
#################################################################################################


################################## Graph Convolution ############################################
# Use Graph Convolutional Layers
model = GCN(hidden_channels=n_hidden_dim)
print(model)
# Define Adam as optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
# Define loss criterion.
criterion = torch.nn.CrossEntropyLoss()
def train():
    # Training mode
    model.train()
    # Clear gradients.
    optimizer.zero_grad()
    # Perform a single forward pass for this iteration, note that edges are also an argument here
    out = model(data.x, data.edge_index)
    # Compute the loss solely based on the training nodes
    loss = criterion(out[data.train_mask],
                     data.y[data.train_mask])
    # Derive gradients.
    loss.backward()
    # Update parameters based on gradients.
    optimizer.step()
    return loss

def test():
    # Evaluation mode
    model.eval()
    # Compute predictions as output of model
    out = model(data.x, data.edge_index)
    # Use the class with highest probability since the output is a vector of softmax
    pred = out.argmax(dim=1)
    # Compute accuracy as the counter of  prediction vs ground-truth labels.
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    # Derive ratio of correct predictions. to get Accuracy
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    return test_acc

for epoch in range(1, 101):
    loss = train()
    #print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

test_acc_gcn = test()
print(f'Test Accuracy with GCN: {test_acc_gcn:.4f}')
#################################################################################################



######################## How a GCN Layer looks inside ? ###############################
#https://dsgiitr.com/blogs/gcn/
import torch.nn as nn
import torch.optim as optim
class GCNConv(nn.Module):
    def __init__(self, A, in_channels, out_channels):
        super(GCNConv, self).__init__()
        self.A_hat = A + torch.eye(A.size(0))
        self.D = torch.diag(torch.sum(A, 1))
        self.D = self.D.inverse().sqrt()
        self.A_hat = torch.mm(torch.mm(self.D, self.A_hat), self.D)
        self.W = nn.Parameter(torch.rand(in_channels, out_channels))

    def forward(self, X):
        out = torch.relu(torch.mm(torch.mm(self.A_hat, X), self.W))
        return out

########################################################################################