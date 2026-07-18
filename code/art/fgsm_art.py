import torch.nn as nn

# prosty model CNN (jedna warstwa linearna dla MNIST)
model = nn. Sequential(
         nn.Flatten(),

         nn.Linear(28*28, 10)
)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


# dummy dane testowe (10 obrazów 28x28)
x_test = torch.rand((10, 1, 28, 28))
y_test = torch.randint(0, 10, (10,))

# konwersja na NumPy (ART wymaga)
x_test_np = x_test.numpy()
y_test_np = y_test.numpy()

from art.estimators.classification import PyTorchClassifier
classifier = PyTorchClassifier(model=model, loss=loss_fn, optimizer=optimizer,
    				  input_shape=(1, 28, 28), nb_classes=10)
 
