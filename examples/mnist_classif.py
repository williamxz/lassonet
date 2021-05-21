from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from keras.datasets import cifar10

from lassonet import LassoNetClassifier

def process_arr(X):
    X = X.reshape(-1, 32,32,3)
    X = X.mean(axis=-1)
    X = X.reshape(-1, 32*32)/255
    return X
    
def load_cifar(dataSize=None):
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    if dataSize is None:
        dataSize = X_train.shape[0]
    X_train, X_test = process_arr(X_train)[:dataSize], process_arr(X_test)
    y_train, y_test = y_train.squeeze()[:dataSize], y_test.squeeze()
    return X_train, y_train, X_test, y_test

'''
X, y = fetch_openml(name="mnist_784", return_X_y=True)
#filter = y.isin(["5", "6"])
X = X.values / 255 #[filter].values / 255
y = LabelEncoder().fit_transform(y)#[filter])

X_train, X_test, y_train, y_test = train_test_split(X, y)
'''
X_train, y_train, X_test, y_test = load_cifar()

model = LassoNetClassifier(M=30, hidden_dims=(16,512), verbose=True)
path = model.path(X_train, y_train)

img = model.feature_importances_.reshape(32, 32)

plt.title("Feature importance")
plt.imshow(img)
plt.colorbar()
plt.savefig("cifar-classification-importance.png")

n_selected = []
accuracy = []
lambda_ = []

for save in path:
    model.load(save.state_dict)
    y_pred = model.predict(X_test)
    n_selected.append(save.selected.sum())
    accuracy.append(accuracy_score(y_test, y_pred))
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
    img = (weight[1] - weight[0]).reshape(32, 32)
    plt.imshow(img)
    plt.colorbar()
    plt.savefig(f"mnist-classification-{i}.png")

fig = plt.figure(figsize=(12, 12))

n_selected = [k.cpu() for k in n_selected]
plt.subplot(311)
plt.grid(True)
plt.plot(n_selected, accuracy, ".-")
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

plt.savefig("mnist-classification-training.png")
