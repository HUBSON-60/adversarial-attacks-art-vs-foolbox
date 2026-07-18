import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import foolbox as fb
import matplotlib.pyplot as plt

print("Biblioteki załadowane!")


class SimpleMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(28 * 28, 10)  # bardzo prosty model

    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)


model = SimpleMNIST()
model.eval()  # Foolbox wymaga modelu w trybie ewaluacji


transform = transforms.ToTensor()

dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# Pobranie pierwszych 5 obrazków
images = torch.stack([dataset[i][0] for i in range(5)])
labels = torch.tensor([dataset[i][1] for i in range(5)])

print("Obrazki i etykiety wczytane!")
print("images.shape:", images.shape)
print("labels:", labels)
