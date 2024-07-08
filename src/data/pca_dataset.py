import torch, numpy
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
from torch.utils.data import TensorDataset

def get_pca_dataset(train_dataset, test_dataset, n_components):
    train_x = torch.stack([train_dataset[i][0].view(-1) for i in range(len(train_dataset))])
    train_y = torch.tensor([train_dataset[i][1] for i in range(len(train_dataset))])
    test_x = torch.stack([test_dataset[i][0].view(-1) for i in range(len(test_dataset))])
    test_y = torch.tensor([test_dataset[i][1] for i in range(len(test_dataset))])

    train_x = train_x.numpy()
    test_x = test_x.numpy()

    scaler = StandardScaler()
    train_x_scaled = scaler.fit_transform(train_x)
    test_x_scaled = scaler.fit_transform(test_x)

    pca = PCA(n_components=n_components)
    pca = pca.fit(train_x_scaled)

    train_x_pca = pca.transform(train_x_scaled)
    test_x_pca = pca.transform(test_x_scaled)

    train_x_pca = torch.from_numpy(train_x_pca)
    test_x_pca = torch.from_numpy(test_x_pca)
    return TensorDataset(train_x_pca, train_y), TensorDataset(test_x_pca, test_y)