
################################### Libraries ##################################################
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import networkx as nx
import os
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from torch_geometric.utils.convert import to_networkx, from_networkx
import torch
#################################################################################################


############################################ Data Collection ###################################
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

X = np.array(X,dtype=int) # Feature BOW for each document
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


#Build the graph with Networkx
G = nx.Graph()
G.add_nodes_from(nodes)
G.add_edges_from(edge_list)

#obtain the adjacency matrix (A)
A = nx.adjacency_matrix(G)
print('Graph info: ', nx.info(G))


#################################################################################################



##################################### Data Preparation ##########################################




# Prepare Training Validation and Test folders , or CV


# Collect labels and idx in a balanced way
def balanced_sampling_idx(labels, limit=20, val_num=500, test_num=1000):
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


train_idx, val_idx, test_idx = balanced_sampling_idx(labels)
# Pair labels with idx of train val test
y_train =  [ i for n,i in enumerate(labels) if n in train_idx ]
y_val   =  [ i for n,i in enumerate(labels) if n in val_idx ]
y_test  =  [ i for n,i in enumerate(labels) if n in test_idx ]




# Train and Val


# Final Test





#################################################################################################




