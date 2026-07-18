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
        self.fc = nn.Linear(28 * 28, 10)

    def forward(self, x):
        x = self.flatten(x)
        return self.fc(x)


model = SimpleMNIST()
model.eval()


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


# Wrapper Foolbox
fmodel = fb.PyTorchModel(model, bounds=(0, 1))


# Atak FGSM
attack = fb.attacks.FGSM()

adv_images_list = attack(
    fmodel,
    images,
    labels,
    epsilons=0.3
)

adv_images = adv_images_list[0]


print("Atak FGSM wykonany!")


# Konwersja do numpy
images_np = images.numpy()
adv_images_np = adv_images.numpy()


# Wizualizacja
plt.figure(figsize=(10, 2))

for i in range(5):

    # Oryginalny obraz
    plt.subplot(2, 5, i + 1)
    plt.imshow(images_np[i].squeeze(), cmap="gray")
    plt.axis("off")

    if i == 0:
        plt.title("Oryginalny")


    # Obraz po ataku
    plt.subplot(2, 5, i + 6)
    plt.imshow(adv_images_np[i].squeeze(), cmap="gray")
    plt.axis("off")

    if i == 0:
        plt.title("Adversarial")
plt.show()
