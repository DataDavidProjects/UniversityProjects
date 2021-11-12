import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.utils.data as data_utils

import sklearn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.pipeline import Pipeline


# Generate simulated data
n = 10000
m = 100
data  = make_classification(n_samples=n,
                    n_features=m,
                    n_informative=int(m*0.5),
                    n_redundant=int(1-(m*0.5)),
                    n_classes=2)

# Numpy features and target
X = data[0]
y = data[1]
# Pandas dataframe
df = pd.DataFrame(X ,columns=[f'x{i}' for i in range(1,m+1)]).join(pd.DataFrame(y,columns=['class']))

# Tensor numpy
tensor_features = torch.from_numpy(X)
tensor_target = torch.from_numpy(y)

# Make everything on the same scale for PCA/Autoencoders
scaler = StandardScaler().fit(X=X)
X_scaled = scaler.transform(X=X)

# Map the output of train test split to make tensor
X_train_t , X_test_t , y_train_t, y_test_t  = map(torch.from_numpy,train_test_split(X_scaled,y,test_size=0.3))


# =============================== DENOISING AUTOENCODER ===========================================
class Encoder(nn.Module):
    def __init__(self, encoded_space_dim, input_dim):
        super().__init__()
        # Encoding
        self.encoder = nn.Sequential(
            nn.Linear(input_dim,encoded_space_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x

class Decoder(nn.Module):
    def __init__(self, encoded_space_dim, input_dim):
        super().__init__()
        # Dencoding
        self.decoder = nn.Sequential(
            nn.Linear(encoded_space_dim, input_dim)
        )

    def forward(self, x):
        x = self.decoder(x)
        return x

# Loss Function Defintion
loss_fn = torch.nn.MSELoss()
# Learning Rate
lr= 0.001
# Batch size
minibatch_size = 2**10
# Dimension reduced
encoded_space_dim = int(m*0.5)


# Init Encoding and Deconding
encoder = Encoder(encoded_space_dim=encoded_space_dim,input_dim=m)
decoder = Decoder(encoded_space_dim=encoded_space_dim,input_dim=m)
# Parameter to optimize for Optimizer obj
params_to_optimize = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
]
# Optimizer
optimizer = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)

# Move both the encoder and the decoder to the selected device CPU or GPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')
encoder.to(device)
decoder.to(device)

# Adding Noise to Original Input in order to reconstruct data
# TO FIX expected scalar type Float but found Double
def add_noise(inputs, noise_factor=0.3):
    # Add to inputs normal noise
    noisy = inputs + torch.randn_like(inputs) * noise_factor
    noisy_scaled = torch.clip(noisy , min=0 , max=1)
    return noisy_scaled

# Create iterators from batch and DataLoader
train_loader = data_utils.DataLoader(X_train_t.float(), batch_size=minibatch_size, shuffle=False)
test_loader = data_utils.DataLoader(X_test_t.float(), batch_size=1, shuffle=False)


#____________________________________ Model Training ____________________________________
# Save loss/train history
history = {}
history['train_loss'] = []
history['test_loss'] = []
# Train autoencoder for unsupervised

# Noise added to original input
noise_factor=0.30
#Epochs
num_epochs  = 20

# START TRAINING
for epoch in range(num_epochs):
    train_loss = []
    # For each batch in data :
        # Add Noise
        # Encode
        # Decode
    # Set train mode for both the encoder and the decoder
    #encoder.train()
    #decoder.train()
    for data_batch in train_loader:
        # Add noise
        data_noisy = add_noise(data_batch, noise_factor)
        # Put to device cpu
        data_noisy = data_noisy.to(device)
        # ENCODING TRAINING
        encoded_data = encoder(data_noisy)
        # DECODING TRAINING
        decoded_data = decoder(encoded_data)
        # Evaluate loss RECONSTRUCT/Decoded data VS ORIGINAL in batch
        loss = loss_fn(decoded_data, data_batch)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        #print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())

    mean_loss = np.mean(train_loss)
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch + 1, num_epochs, mean_loss))
    history['train_loss'].append(mean_loss)


# Apply Supervised ML with extracted Features
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support

# DENOISE TEST
test_noise = add_noise(X_test_t)
tensor_rec = decoder(encoder(test_noise.float())) - X_test_t
df_rec = pd.DataFrame(tensor_rec.detach().numpy())