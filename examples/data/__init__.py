import os

import numpy as np
import torch
import torchvision.datasets
from sklearn.datasets import fetch_openml
from sklearn.datasets import load_boston, fetch_california_housing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale


def get_boston_housing():
    dataset = load_boston()
    X = dataset.data
    y = dataset.target
    _, true_features = X.shape
    # add dummy feature
    X = np.concatenate([X, np.random.randn(*X.shape)], axis=1)
    feature_names = list(dataset.feature_names) + ["fake"] * true_features

    # standardize
    X = StandardScaler().fit_transform(X)
    y = scale(y)
    return X, y


def get_mnist(filter=None):
    filter = filter or [f"{i}" for i in range(10)]
    X, y = fetch_openml(name="mnist_784", return_X_y=True)
    filter = y.isin(filter)
    X = X[filter].values / 255
    y = LabelEncoder().fit_transform(y[filter])
    return X, y


def get_cifar_small(grayscale=True):
    X, y = fetch_openml(name="cifar_10_small", return_X_y=True)
    if grayscale:
        X = X.reshape(-1, 32, 32, 3)
        X = X.mean(axis=-1)
        X = X.reshape(-1, 32 * 32)
    X = X / 255
    return X, y


def get_cifar(train=True, batch_size=64, val_size=.1, flatten=False):
    if flatten:
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                                     (0.2023, 0.1994, 0.2010)),
                                                    torchvision.transforms.Lambda(lambda x: torch.flatten(x))])
    else:
        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                                     (0.2023, 0.1994, 0.2010))])
    dataset = torchvision.datasets.CIFAR10(os.path.join(os.getcwd(), 'data', 'cifar10'), train=train, download=True,
                                           transform=transform)
    if not train:
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return data_loader
    else:
        trainset, valset = torch.utils.data.random_split(dataset,
                                                         [round((1 - val_size) * len(dataset)),
                                                          round(val_size * len(dataset))])
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True)
        return train_loader, val_loader


def get_california_housing():
    return fetch_california_housing(return_X_y=False)


def get_loader(X, y, batch_size=64, shuffle=True):
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=batch_size, shuffle=shuffle)
