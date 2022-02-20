"""
The main idea is to demonstrate how Embedding a Graph Structures outperform
traditional machine learning preprocessing.

One graph structure which is

* Static
* Semisupervised Context
* Definition of a Benchamrk using CORA
*
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import networkx as nx
import os
from sklearn.utils import shuffle
from sklearn.metrics import classification_report

all_data = []
all_edges = []

path  = "C:/Users/david/Desktop/Graph_Convolutional_Networks_Node_Classification-master/Graph_Convolutional_Networks_Node_Classification-master/cora"

for root, dirs, files in os.walk(path):
    for file in files:
        if '.content' in file:
            with open(os.path.join(root, file), 'r') as f:
                all_data.extend(f.read().splitlines())
        elif 'cites' in file:
            with open(os.path.join(root, file), 'r') as f:
                all_edges.extend(f.read().splitlines())

# Shuffle the data because the raw data is ordered based on the label
random_state = 77
all_data = shuffle(all_data, random_state=random_state)


#parse the data
labels = []
nodes = []
X = []

for i,data in enumerate(all_data):
    elements = data.split('\t')
    labels.append(elements[-1])
    X.append(elements[1:-1])
    nodes.append(elements[0])

X = np.array(X,dtype=int)
N = X.shape[0] #the number of nodes
F = X.shape[1] #the size of node features
print('X shape: ', X.shape)


#parse the edge
edge_list=[]
for edge in all_edges:
    e = edge.split('\t')
    edge_list.append((e[0],e[1]))

print('\nNumber of nodes (N): ', N)
print('\nNumber of features (F) of each node: ', F)
print('\nCategories: ', set(labels))

num_classes = len(set(labels))
print('\nNumber of classes: ', num_classes)


def limit_data(labels, limit=20, val_num=500, test_num=1000):
    '''
    Get the index of train, validation, and test data
    '''
    label_counter = dict((l, 0) for l in labels)
    train_idx = []

    for i in range(len(labels)):
        label = labels[i]
        if label_counter[label] < limit:
            # add the example to the training data
            train_idx.append(i)
            label_counter[label] += 1

        # exit the loop once we found 20 examples for each class
        if all(count == limit for count in label_counter.values()):
            break

    # get the indices that do not go to traning data
    rest_idx = [x for x in range(len(labels)) if x not in train_idx]
    # get the first val_num
    val_idx = rest_idx[:val_num]
    test_idx = rest_idx[val_num:(val_num + test_num)]
    return train_idx, val_idx, test_idx


train_idx, val_idx, test_idx = limit_data(labels)
# Pair labels with idx of train val test
y_train =  [ i for n,i in enumerate(labels) if n in train_idx ]
y_val   =  [ i for n,i in enumerate(labels) if n in val_idx ]
y_test  =  [ i for n,i in enumerate(labels) if n in test_idx ]


#set the mask
train_mask = np.zeros((N,),dtype=bool)
train_mask[train_idx] = True

val_mask = np.zeros((N,),dtype=bool)
val_mask[val_idx] = True

test_mask = np.zeros((N,),dtype=bool)
test_mask[test_idx] = True

#Build the graph with Networkx
G = nx.Graph()
G.add_nodes_from(nodes)
G.add_edges_from(edge_list)

#obtain the adjacency matrix (A)
A = nx.adjacency_matrix(G)
print('Graph info: ', nx.info(G))


# Cast as tensors
train_mask,test_mask,val_mask = map(torch.tensor,[train_mask,test_mask, val_mask])
A = torch.tensor(A.todense(), dtype=torch.float32)
X = torch.tensor(X, dtype=torch.float32)
# Graph Convolutional Layer
class GCNConv(nn.Module):
    def __init__(self, A, in_channels, out_channels):
        super(GCNConv, self).__init__()
        # Self loop on Adjacency Matrix
        self.A_hat = A + torch.eye(A.size(0))
        # Degree matrix
        self.D = torch.diag(torch.sum(A, 1))
        # Inverse square root  of Degree Matrix
        self.D = self.D.inverse().sqrt()
        # Adjacency Matrix with self loop and Symmetric Normalization D^-0.5A'D^-0.5
        self.A_hat = torch.mm(torch.mm(self.D, self.A_hat), self.D)
        # Random Initialization
        self.W = nn.Parameter(torch.rand(in_channels, out_channels,dtype=torch.float32))

    def forward(self, X):
        out = torch.relu(torch.mm(torch.mm(self.A_hat, X), self.W))
        return out


class Graph_Net(torch.nn.Module):
    def __init__(self, A, n_features, hidden_channels, num_classes):
        super(Graph_Net, self).__init__()
        self.conv1 = GCNConv(A, n_features, hidden_channels)
        self.out = nn.Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        # First Message Passing Layer (Transformation)
        x = self.conv1(x, edge_index)
        # Output layer from logits to probabilities with Softmax
        x = F.softmax(self.out(x), dim=1)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
hidden_channels = 16
model = Graph_Net(A=A,n_features=F,
                  hidden_channels=hidden_channels,
                  num_classes= num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)



criterion = nn.CrossEntropyLoss()
def train():
    model.train()
    optimizer.zero_grad()
    # Use all data as input, because all nodes have node features
    out = model(X,edge_list)
    # Only use nodes with labels available for loss calculation --> mask
    loss = criterion(out[train_mask], y_train)
    loss.backward()
    optimizer.step()
    return loss

losses = []
for epoch in range(0, 1001):
    loss = train()
    losses.append(loss)
    if epoch % 100 == 0:
      print(f'Epoch: {epoch:03d}  Loss: {loss:.4f}')