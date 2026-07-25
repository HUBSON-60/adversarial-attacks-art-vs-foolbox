import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod


# Prosty model klasyfikacyjny dla MNIST
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 10)
)

model.eval()


# Funkcja straty i optymalizator wymagane przez ART
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# Wczytanie zbioru MNIST
transform = transforms.ToTensor()

dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)


# Pobranie 10 obrazów testowych
x_test = torch.stack([dataset[i][0] for i in range(10)])
y_test = torch.tensor([dataset[i][1] for i in range(10)])


print("MNIST załadowany!")
print("x_test shape:", x_test.shape)
print("y_test:", y_test)


# ART wymaga danych w formacie NumPy
x_test_np = x_test.numpy()
y_test_np = y_test.numpy()


# Utworzenie klasyfikatora ART
classifier = PyTorchClassifier(
    model=model,
    loss=loss_fn,
    optimizer=optimizer,
    input_shape=(1, 28, 28),
    nb_classes=10
)


# Atak FGSM
attack = FastGradientMethod(
    estimator=classifier,
    eps=0.3
)


# Generowanie przykładów adwersarialnych
x_test_adv = attack.generate(
    x=x_test_np
)


print("FGSM wykonany!")
print("x_test_adv shape:", x_test_adv.shape)
