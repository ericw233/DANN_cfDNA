import numpy as np
import pandas as pd
import os
import re
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt


# Load Dec2023 feature
feature_path = '/mnt/binf/eric/Mercury_Dec2023/Feature/'
feature_files = os.listdir(feature_path)
pattern = re.compile(r'(All)|(gemini)')
feature_files_select = [file_tmp for file_tmp in feature_files if pattern.search(file_tmp)]

feature_data = pd.DataFrame()
for file_tmp in feature_files_select:
    if file_tmp.endswith(".csv"):
        data_tmp = pd.read_csv(feature_path + file_tmp)
    if feature_data.size > 0:
        feature_data = pd.merge(feature_data, data_tmp, on="SampleID",how="inner")
    else:
        feature_data = data_tmp

print(f"The shape of feature data is {feature_data.shape}")

sample_info = pd.read_csv('/mnt/binf/eric/Mercury_Dec2023/Info/Test1.all.full.info.list', sep='\t')
feature_data_annotate = pd.merge(feature_data, sample_info.loc[:,["SampleID","Train_Group"]], on="SampleID", how="inner")
mapping = {'Healthy':0,'Cancer':1}

X = feature_data_annotate.drop(columns=["SampleID","Train_Group"])
y = feature_data_annotate["Train_Group"].replace(mapping)
sampleid = feature_data_annotate["SampleID"]

# Standardize the data
scaler = StandardScaler()
X_std = scaler.fit_transform(X)


# Split the data into training and testing sets
X_train, X_test, sampleid_train, sampleid_test = train_test_split(X_std, sampleid, test_size=0.2, random_state=42)

mean_imputer = SimpleImputer(strategy = 'mean')
X_train_imputed = mean_imputer.fit_transform(X_train)
X_test_imputed = mean_imputer.transform(X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.Tensor(X_train_imputed)
X_test_tensor = torch.Tensor(X_test_imputed)

# Autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self, input_size, encoding_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, encoding_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_size, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_size)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Instantiate the autoencoder
input_size = X_train_imputed.shape[1]
encoding_size = 64
autoencoder = Autoencoder(input_size, encoding_size)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = Adam(autoencoder.parameters(), lr=0.001)

# Training the autoencoder
num_epochs = 256
for epoch in range(num_epochs):
    outputs = autoencoder(X_train_tensor)
    loss = criterion(outputs, X_train_tensor)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Encode data using the trained autoencoder
encoded_test = autoencoder.encoder(X_test_tensor).detach().numpy()
encoded_train = autoencoder.encoder(X_train_tensor).detach().numpy()

# Apply K-Means clustering on the encoded data
kmeans = KMeans(n_clusters=4, random_state=42)
clusters_train = kmeans.fit_predict(encoded_train)
clusters_test = kmeans.predict(encoded_test)

sampleid_cluster_df = pd.DataFrame({'SampleID': np.concatenate((sampleid_train, sampleid_test)),
                                   'Cluster': np.concatenate((clusters_train, clusters_test))})

sample_info_annotated = pd.merge(sample_info, sampleid_cluster_df, on="SampleID", how="inner")
table1 = pd.crosstab(sample_info_annotated["Train_Group"],sample_info_annotated["Cluster"])
table2 = pd.crosstab(sample_info_annotated["ProjectID"],sample_info_annotated["Cluster"])

sample_info_annotated.to_csv("/mnt/binf/eric/Mercury_Dec2023/Sampleinfo_Cluster.csv",index=False)



# Visualize the results
plt.scatter(encoded_test[:, 0], encoded_test[:, 1], c=clusters_test, cmap='viridis')
plt.title('Autoencoder-based Clustering: test set')
plt.xlabel('Encoded Feature 1')
plt.ylabel('Encoded Feature 2')

plt.scatter(encoded_train[:, 0], encoded_train[:, 1], c=clusters_train, cmap='viridis')
plt.title('Autoencoder-based Clustering: train set')
plt.xlabel('Encoded Feature 1')
plt.ylabel('Encoded Feature 2')

