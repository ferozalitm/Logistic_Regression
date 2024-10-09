# Logistic regression /Classification using MNIST serialized images
import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

torch.manual_seed(0)

# Hyperparameters
ip_dimn = 28*28
no_classes = 10
batch_size = 256

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = torchvision.datasets.MNIST(root='data/',
                                     train=True, 
                                     transform=transforms.ToTensor(),
                                     download=True)

test_dataset = torchvision.datasets.MNIST(root='data/',
                                     train=False, 
                                     transform=transforms.ToTensor(),
                                     download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size, 
                                           shuffle=False)

no_batches_train = len(train_loader)
no_batches_tst = len(test_loader)
print(f"No_batches train: {no_batches_train}")
print(f"No_batches test: {no_batches_tst}")

# Build a fully connected layer and forward pass
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(in_features=ip_dimn, out_features=14*14)
        self.linear2 = nn.Linear(in_features=14*14, out_features=7*7)
        self.linear3 = nn.Linear(in_features=7*7, out_features=no_classes)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.linear1(x))
        x = self.relu2(self.linear2(x))
        x = self.linear3(x)
        return x

# Build model.
model = Net().to(device)

# Build optimizer.
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  

# Build loss.
criterion = nn.CrossEntropyLoss()

no_epochs = 100
first_pass = True

for epoch in range(no_epochs):

  # Training
  batch_idx = 0
  total_loss = 0
  total_correct = 0

  for batch_idx, (images, labels) in enumerate(train_loader):

    images = images.reshape(-1, 28*28)
    images = images.to(device)
    labels = labels.to(device)

    # Forward pass.
    pred = model(images)

    # Compute loss.
    loss = criterion(pred, labels)
    if epoch == 0 and first_pass == True:
      print(f'Initial {epoch} loss: ', loss.item())
      first_pass = False

    # Compute gradients.
    optimizer.zero_grad()
    loss.backward()

    # 1-step gradient descent.
    optimizer.step()

    # calculating train loss
    total_loss += loss.item()
    total_correct += torch.sum(labels == torch.argmax(pred, dim=1))

    if epoch == 0 and (batch_idx+1) % 10 == 0:
      print(f"Train Batch:{batch_idx}/{no_batches_train}, loss: {loss}, total_loss: {total_loss}")

  print(f'Train Epoch:{epoch}, Average Train loss:{total_loss/no_batches_train}, Average Train accuracy:{total_correct/len(train_dataset)*100.} ', )

  # Testing after each epoch
  model.eval()
  with torch.no_grad():

    total_loss = 0
    total_correct = 0

    for images, labels in test_loader:

      images = images.reshape(-1, 28*28)
      images = images.to(device)
      labels = labels.to(device)

      # Forward pass.
      pred = model(images)

      # Compute test loss.
      loss = criterion(pred, labels)
      total_loss += loss.item()
      total_correct += torch.sum(labels == torch.argmax(pred, dim=1))
      # print(f"test Batch:{batch_idx}/{len(test_loader)}, loss: {loss}, total_loss: {total_loss}")

    print(f'Test Epoch:{epoch}, Average Test loss: {total_loss/no_batches_tst}, Average Test accuracy: {total_correct/len(test_dataset)*100.}', )

  model.train()
