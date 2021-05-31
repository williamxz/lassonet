import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from data import get_mnist, get_cifar
from lassonet import LassoNetClassifier
import os
import torch
from itertools import islice
import random
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader, val_loader, test_loader = get_cifar(train=True, batch_size=64, val_size=.1, flatten=True)
print("Data Loaded.")
dims = [(100,), (8, 100), (20, 100), (80, 100), (8, 80, 100), (20, 80, 100), (80, 80, 100), (8, 20, 100), (20, 20, 100),
        (80, 20, 100)]

num_ftrs = 32 * 32 * 3


class Model(torch.nn.Module):
    def __init__(self, *dims):
        """
        first dimension is input
        last dimension is output
        """
        assert len(dims) > 2
        super().__init__()

        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self.skip = torch.nn.Linear(dims[0], dims[-1], bias=False)

    def forward(self, inp):
        current_layer = inp
        for theta in self.layers:
            current_layer = theta(current_layer)
            if theta is not self.layers[-1]:
                current_layer = torch.nn.functional.relu(current_layer)
        return current_layer


def evaluate(model, data, mask=None):
    model.eval()
    y_pred = []
    y_true = []
    for X, y in data:
        X, y = X.to(device), y.to(device)
        if mask is not None:
            X = X * mask
        y_pred.extend(model(X).tolist())
        y_true.extend(y.tolist())
        return y_pred, y_true


def train(model, epochs, data, iterationsPerEpoch, criterion, optimizer, patience, mask=None):
    train_data, val_data = data
    pred, true = evaluate(model, val_data, mask)
    best_obj = criterion(torch.tensor(pred).to(device),
                         torch.tensor(true).to(device)).item()
    epochs_since_best_obj = 0

    for i in range(epochs):
        model.train()
        enumeratedLoader = enumerate(train_data) if iterationsPerEpoch < 0 else islice(enumerate(train_data), 0,
                                                                                       iterationsPerEpoch)
        for batch_num, (X, y) in enumeratedLoader:
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            if mask is not None:
                X = X * mask
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()

        pred, true = evaluate(model, val_data, mask)
        obj = criterion(torch.tensor(pred).to(device),
                        torch.tensor(true).to(device)).item()
        if obj < .99 * best_obj:
            best_obj = obj
            epochs_since_best_obj = 0
        else:
            epochs_since_best_obj += 1
        if patience is not None and epochs_since_best_obj >= patience:
            break


for dim in dims:
    print(dim)
    lasso_acc = []
    base_acc = []
    random_acc = []
    for sample in test_loader:
        X, y = sample
        break
    standard_model = Model(X.shape[1],
                           *dim,
                           (y.max() + 1).item(), ).to(device)

    epochs = (1000, 100)
    iterationsPerEpoch = 100
    patience = (10, 5)
    model = LassoNetClassifier(M=30, verbose=True, hidden_dims=dim, n_iters=epochs, patience=patience,
                               lambda_start=5e2, path_multiplier=1.03)
    path = model.path((train_loader, val_loader), stochastic=True, verboseEpochs=False,
                      iterationsPerEpoch=iterationsPerEpoch)

    for save in path:
        model.load(save.state_dict)
        y_pred, y_true = model.predict(test_loader, stochastic=True)
        save.acc = accuracy_score(y_true, y_pred)

    path = sorted(path, key=lambda k: k.acc, reverse=True)
    to_study = [.01, .1]
    for save in path:
        if not to_study:
            break
        if save.selected.sum() > to_study[-1] * num_ftrs:
            continue
        to_study.pop()
        lasso_acc.append(save.acc)

        mask = save.selected.to(device)
        standard_model = Model(X.shape[1],
                               *dim,
                               (y.max() + 1).item(), ).to(device)
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        optimizer = torch.optim.Adam(standard_model.parameters(), lr=1e-3)
        train(standard_model, epochs[1], (train_loader, val_loader), iterationsPerEpoch, criterion, optimizer,
              patience[1], mask=mask)
        y_pred, y_true = evaluate(standard_model, test_loader, mask=mask)
        base_acc.append(accuracy_score(y_true, np.argmax(y_pred, axis=1)))

        acc = 0
        for i in range(5):
            new_mask = torch.zeros_like(mask).to(device)
            mask_idx = random.sample(list(range(num_ftrs)), mask.sum().item())
            new_mask[mask_idx] = 1.
            standard_model = Model(X.shape[1],
                                   *dim,
                                   (y.max() + 1).item(), ).to(device)
            criterion = torch.nn.CrossEntropyLoss(reduction="mean")
            optimizer = torch.optim.Adam(standard_model.parameters(), lr=1e-3)
            train(standard_model, epochs[1], (train_loader, val_loader), iterationsPerEpoch, criterion, optimizer,
                  patience[1], mask=new_mask)
            y_pred, y_true = evaluate(standard_model, test_loader, mask=new_mask)
            acc += accuracy_score(y_true, np.argmax(y_pred, axis=1))
        random_acc.append(acc / 5)

    with open(os.path.join(os.getcwd(), 'selector_log.txt'), 'a') as log:
        log.write(str(dim)+'\n')
        log.write('lasso: ' + str(lasso_acc)+'\n')
        log.write('base: ' + str(base_acc)+'\n')
        log.write('rando: ' + str(random_acc)+'\n')
        log.write('='*20+'\n')
