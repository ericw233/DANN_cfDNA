import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, DBSCAN, MeanShift
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


def AE_clustering(X_input, sampleid_input, methods = "kmeans", encoding_size = 64, n_cluster = 4):

    # X_input is a NA-droped, standardized, and mean-imputed data frame, not a tensor
    X_train, X_test, sampleid_train, sampleid_test = train_test_split(X_input, sampleid_input, test_size=0.2, random_state=42)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    
    # Autoencoder architecture
    class Autoencoder(nn.Module):
        def __init__(self, input_size, encoding_size):
            super(Autoencoder, self).__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_size, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, encoding_size)
            )
            self.decoder = nn.Sequential(
                nn.Linear(encoding_size, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, input_size)
            )

        def forward(self, x):
            x = self.encoder(x)
            x = self.decoder(x)
            return x

    ### finish defining AE structure
       
    
    # Instantiate the autoencoder
    input_size = X_train.shape[1]
    # encoding_size = 64
    autoencoder = Autoencoder(input_size, encoding_size)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = Adam(autoencoder.parameters(), lr=1e-3, weight_decay=1e-4)

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

    if methods == "kmeans":
        # Apply K-Means clustering on the encoded data
        kmeans = KMeans(n_clusters=n_cluster, random_state=42)
        clusters_train = kmeans.fit_predict(encoded_train)
        clusters_test = kmeans.predict(encoded_test)

    elif methods == "DBSCAN":
        dbscan = DBSCAN(eps=0.5, min_samples=20)
        clusters_train = dbscan.fit_predict(encoded_train)
        clusters_test = dbscan.predict(encoded_test)
       
    elif methods == "GMM":
        gmm = GaussianMixture(eps=0.5, min_samples=5)
        clusters_train = gmm.fit_predict(encoded_train)
        clusters_test = gmm.predict(encoded_test)
    
    elif methods == "MeanShift":
        meanshift = MeanShift(eps=0.5, min_samples=5)
        clusters_train = meanshift.fit_predict(encoded_train)
        clusters_test = meanshift.predict(encoded_test)
        
        
    sampleid_cluster_df = pd.DataFrame({'SampleID': np.concatenate((sampleid_train, sampleid_test)),
                                    'Cluster': np.concatenate((clusters_train, clusters_test))})

    
    # Visualize the results
    plt.scatter(encoded_test[:, 0], encoded_test[:, 1], c=clusters_test, cmap='viridis')
    plt.title('Autoencoder-based Clustering: test set')
    plt.xlabel('Encoded Feature 1')
    plt.ylabel('Encoded Feature 2')

    plt.scatter(encoded_train[:, 0], encoded_train[:, 1], c=clusters_train, cmap='viridis')
    plt.title('Autoencoder-based Clustering: train set')
    plt.xlabel('Encoded Feature 1')
    plt.ylabel('Encoded Feature 2')
    
    return sampleid_cluster_df




