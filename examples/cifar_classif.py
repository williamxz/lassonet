import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from data import get_mnist, get_cifar
from lassonet import LassoNetClassifier

train_loader, val_loader, test_loader = get_cifar(train=True, batch_size=64, val_size=.1, flatten=True)
print("Data Loaded.")
model = LassoNetClassifier(M=30, verbose=True, hidden_dims=(80, 100), n_iters=(1000, 100), patience=(10,5), lambda_start=5e2, path_multiplier=1.02)
path = model.path((train_loader, val_loader), stochastic=True, verboseEpochs=True, iterationsPerEpoch=100)

img = model.feature_importances_.reshape(3, 32, 32).mean(0)
img = (img * 1.0 / (img.max()))

plt.title("Feature importance.")
plt.imshow(img)
plt.colorbar()
plt.savefig("cifar-classification-importance.png")

n_selected = []
accuracy = []
lambda_ = []

for save in path:
    model.load(save.state_dict)
    y_pred, y_true = model.predict(test_loader, stochastic=True)
    n_selected.append(save.selected.sum())
    accuracy.append(accuracy_score(y_true, y_pred))
    lambda_.append(save.lambda_)

to_plot = [160, 220, 300]

for i, save in zip(n_selected, path):
    if not to_plot:
        break
    if i > to_plot[-1]:
        continue
    to_plot.pop()
    plt.clf()
    plt.title(f"Linear model with {i} features")
    weight = save.state_dict["skip.weight"]
    img = (weight[1] - weight[0]).reshape(32, 32, 3)
    plt.imshow(img)
    plt.colorbar()
    plt.savefig(f"cifar-classification-{i}.png")

fig = plt.figure(figsize=(12, 12))

n_selected = [k.cpu() for k in n_selected]
n_selected_sorted, accuracy_sorted = zip(*sorted(zip(n_selected, accuracy), key = lambda k: k[0]))

plt.subplot(311)
plt.grid(True)
plt.plot(n_selected_sorted, accuracy_sorted, ".-")
plt.xlabel("number of selected features")
plt.ylabel("classification accuracy")

plt.subplot(312)
plt.grid(True)
plt.plot(lambda_, accuracy, ".-")
plt.xlabel("lambda")
plt.xscale("log")
plt.ylabel("classification accuracy")

plt.subplot(313)
plt.grid(True)
plt.plot(lambda_, n_selected, ".-")
plt.xlabel("lambda")
plt.xscale("log")
plt.ylabel("number of selected features")

plt.savefig("cifar-classification-training.png")
