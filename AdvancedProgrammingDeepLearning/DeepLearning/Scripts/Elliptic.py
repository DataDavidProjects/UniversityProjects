import pandas as pd
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

# Path
root = 'C:/Users/david/Desktop/UniversityProjects/AdvancedProgrammingDeepLearning/DeepLearning/Data/'

# General Preprocessing
df_classes = pd.read_csv(root+"elliptic_txs_classes.csv")
df_edges = pd.read_csv(root+"elliptic_txs_edgelist.csv")
df_features = pd.read_csv(root+"elliptic_txs_features.csv", header=None)

colNames1 = {'0': 'txId', 1: "Time step"}
colNames2 = {str(ii+2): "Local_feature_" + str(ii+1) for ii in range(93)}
colNames3 = {str(ii+95): "Aggregate_feature_" + str(ii+1) for ii in range(72)}

colNames = dict(colNames1, **colNames2, **colNames3 )
colNames = {int(jj): item_kk for jj,item_kk in colNames.items()}

df_features = df_features.rename(columns=colNames)

df_classes.loc[df_classes['class'] == 'unknown', 'class'] = 3
# print('Shape of classes', df_classes.shape)
# print('Shape of edges', df_edges.shape)
# print('Shape of features', df_features.shape)

# Merge Class and features
df_class_feature = pd.merge(df_classes, df_features )
df_class_feature.head()

#backup total df
df = df_class_feature.copy()

# Select just the transaction with known outcome
selected_ids = df_class_feature.loc[(df_class_feature['class'] != 3), 'txId']
df_edges_selected = df_edges.loc[df_edges['txId1'].isin(selected_ids)]
df_classes_selected = df_classes.loc[df_classes['txId'].isin(selected_ids)]
df_features_selected = df_features.loc[df_features['txId'].isin(selected_ids)]

# Merge Class and features
df_class_feature_selected = pd.merge(df_classes_selected, df_features_selected )
df_class_feature_selected.head()


############## Approach ##########################
""" 
Option1: We can use a denoiser autoencoder to upsample denoiser and go for a more robust supervised learning
Option2 : Unsupervised Learning on score 
Option3: Ensemble the 2 approches 
"""


#______________________ Preprocessing for Models _______________________________________

" I have to seperate data in frauds and nonfrauds , need to use only non frauds! "
df = df_class_feature_selected.copy()

# Seperate features and labels
X = df.drop(columns=['txId', 'class', 'Time step']) # drop class, text id and time step
y = df[['class']]
#Change label 2 is licit 1 illicit
y = y['class'].apply(lambda x: 0 if x == '2' else 1 )

#________________________________________________________________________________________


# Shuffle data train test split
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.25,
                                                    random_state=15)

# Filter train to get only clean transactions
df_train = pd.concat([X_train , y_train] , axis = 1)
blacklist_index =  df_train.loc[ df_train['class'] == 1 , : ].index
df_clean_train = df_train.loc[set(df_train.index)-set(blacklist_index)]

#Training data with just clean transactions
X_train =  df_clean_train.drop('class',axis =1 )
y_train = df_clean_train[['class']]

#_______________________________Autoencoders________________________________________________
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
from torchvision import transforms
import matplotlib.pyplot as plt

#_________________________________ Definition of Autoencoder ________________________
N = X_train.shape[0]
features_vector = X_train.shape[1]
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(features_vector, 100),
            nn.ReLU(),
            nn.Linear(100, 80),
            nn.ReLU(),
            nn.Linear(80, 50),
            nn.ReLU(),
            nn.Linear(50, 30),
            nn.ReLU(),
            nn.Linear(30, 10)
        )

        self.decoder = nn.Sequential(
            nn.Linear(10, 30),
            nn.ReLU(),
            nn.Linear(30, 50),
            nn.ReLU(),
            nn.Linear(50, 80),
            nn.ReLU(),
            nn.Linear(80, 100),
            nn.ReLU(),
            nn.Linear(100, features_vector)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
#__________________________________________________________________________________________



#___________________________ Init Model __________________________________________________
num_epochs = 50
minibatch_size = 2**5
learning_rate = 1e-3

model = Autoencoder().double().cpu()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=learning_rate,
                             weight_decay=10e-05)


# Convert arrays to pytorch tensor train and test
X_train_t = torch.from_numpy(X_train.values)
y_train_t = torch.from_numpy(y_train.values)

X_test_t = torch.from_numpy(X_test.values)
y_test_t = torch.from_numpy(y_test.values)

# Create iterators from batch and DataLoader
train_loader = data_utils.DataLoader(X_train_t, batch_size=minibatch_size, shuffle=True)
test_loader = data_utils.DataLoader(X_test_t, batch_size=1, shuffle=False)
#________________________________________________________________________________________


#____________________________________ Model Training ____________________________________
# Save loss/train history
history = {}
history['train_loss'] = []
history['test_loss'] = []
# Train autoencoder for unsupervised
for epoch in range(num_epochs):
    h = np.array([])
    for data in train_loader:
        # print(type(data))
        # data = Variable(data).cpu()
        # print(type(data))
        # ===================forward=====================
        output = model(data)
        loss = criterion(output, data)
        h = np.append(h, loss.item())

        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # ===================log========================
    mean_loss = np.mean(h)
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch + 1, num_epochs, mean_loss))
    history['train_loss'].append(mean_loss)

# torch.save(model.state_dict(), './autoencoder_net.pth')


#Evaluation
#history['train_loss']
plt.plot(range(num_epochs),history['train_loss'],'ro',linewidth=2.0)
plt.plot(history['train_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train'], loc='upper right');
plt.show()
#________________________________________________________________________________________


#_________________________________________ Try to reconstruct mixed data with Clean and Frauds________________________
pred_losses = {'pred_loss' : []}
model.eval()
with torch.no_grad():
   # test_loss = 0
    for data in test_loader:
        inputs = data
        # print(inputs)
        outputs = model(inputs)
        loss = criterion(outputs, inputs).data.item()
        #print(loss)
        pred_losses['pred_loss'].append(loss)
        #pred_losses = model([y_test.size, y_test])
reconstructionErrorDF = pd.DataFrame(pred_losses)
reconstructionErrorDF['Class'] = y_test

#________________________________________ Observe Distribution of new lecit and new illecit_________________________________________
fig = plt.figure()
ax = fig.add_subplot(111)
normal_error_df = reconstructionErrorDF[(reconstructionErrorDF['Class']== 0) & (reconstructionErrorDF['pred_loss'] < 10)]
_ = ax.hist(normal_error_df.pred_loss.values, bins=10)
ax.set_title('Lecit Reconstruction')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(111)
fraud_error_df = reconstructionErrorDF[(reconstructionErrorDF['Class']== 1) ]
_ = ax.hist(fraud_error_df.pred_loss.values, bins=10)
ax.set_title('Illecit Reconstruction')
plt.show()
#________________________________________________________________________________________________________________________



#_______________________________________ Use the threshoold of the reconstruction error to spot anomalies _______________
groups = reconstructionErrorDF.groupby('Class')
fig, ax = plt.subplots()
threshold = 2.0
for name, group in groups:
    ax.plot(group.index, group.pred_loss, marker='o', ms=3.5, linestyle='',
            label= "Fraud" if name == 1 else "Normal")
ax.hlines(threshold, ax.get_xlim()[0], ax.get_xlim()[1], colors="r", zorder=100, label='Threshold')
ax.legend()
plt.title("Reconstruction error for different classes")
plt.ylabel("Reconstruction error")
plt.xlabel("Data point index")
plt.show();
#________________________________________________________________________________________________________________________