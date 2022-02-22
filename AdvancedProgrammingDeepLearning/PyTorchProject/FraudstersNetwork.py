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
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures , AddSelfLoops
#################################################################################################


from faker import Faker
fake = Faker(['en_US'])
Faker.seed(0)



# fake.iban() + fake.local_latlng()
for _ in range(5):
    print(fake.profile())

n = 100
profiles = pd.DataFrame([ pd.Series(fake.profile()) for i in range(n) ])
print(profiles.head())