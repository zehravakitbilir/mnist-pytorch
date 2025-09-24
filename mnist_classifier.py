import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import random

batch_size = 64
epochs = 10
learning_rate = 0.01

# Datenvorbereitung
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

# Einfaches neuronales Netzwerk
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
 
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Listen zum Speichern
train_losses = []
test_accuracies = []

# Training
for epoch in range(epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    # Durchschnittlicher Loss
    epoch_loss = running_loss / len(train_set)
    train_losses.append(epoch_loss)

    # Test-Accuracy
    correct = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    epoch_accuracy = 100 * correct / total
    test_accuracies.append(epoch_accuracy)
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')
    model.train()

# Grafische Darstellung von Loss und Accuracy
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(range(1, epochs+1), train_losses, marker='o')
plt.title('Train Loss pro Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1,2,2)
plt.plot(range(1, epochs+1), test_accuracies, marker='o', color='orange')
plt.title('Test Accuracy pro Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')

plt.show()

# Zufälliges Testbild und Vorhersage
images, labels = next(iter(test_loader))
idx = random.randint(0, len(images)-1)
img = images[idx]
label = labels[idx]

plt.imshow(img[0], cmap='gray')
plt.title(f'True Label: {label.item()}')
plt.show()

model.eval()
with torch.no_grad():
    output = model(img.unsqueeze(0))
    predicted_label = torch.argmax(output, 1).item()
print(f'Das Netzwerk sagt: {predicted_label}')
