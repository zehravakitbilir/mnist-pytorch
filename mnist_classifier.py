import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import random
import matplotlib.pyplot as plt

batch_size = 64 #anzahl der Bilder pro Batch
epochs = 2 #anzahl der Durchläufe durch den gesamten Datensatz
learning_rate = 0.01 #Lernrate für den Optimierer


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

#Definiert einfaches neuronales Netzwerk
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 128) #Eingabeschicht zu versteckter Schicht
        self.fc2 = nn.Linear(128, 64)    #Versteckte Schicht zu weiterer versteckter Schicht
        self.fc3 = nn.Linear(64, 10)     #Versteckte Schicht zu Ausgabeschicht
 
    def forward(self, x): #Vorwärtsdurchlauf
        x = x.view(-1, 28*28)            #Bilder in Vektorform umwandeln
        x = torch.relu(self.fc1(x))      #ReLU-Aktivierungsfunktion nach erster Schicht
        x = torch.relu(self.fc2(x))      #ReLU-Aktivierungsfunktion nach zweiter Schicht
        x = self.fc3(x)                  #Ausgabeschicht (keine Aktivierung, da CrossEntropyLoss Softmax beinhaltet)
        return x
    
model = SimpleNN()
criterion = nn.CrossEntropyLoss() #Kreuzentropie-Verlustfunktion für Mehrklassenklassifikation
optimizer = optim.SGD(model.parameters(), lr=learning_rate) #Stochastic Gradient Descent

#Training des Modells
for epoch in range(epochs):
    for images, labels in train_loader:
        outputs = model(images)          #Vorwärtsdurchlauf
        loss = criterion(outputs, labels) #Verlust berechnen

        optimizer.zero_grad()            #Gradienten zurücksetzen
        loss.backward()                  #Rückwärtsdurchlauf
        optimizer.step()                 #Gewichte aktualisieren
    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

#Test
correct = 0
total = 0
with torch.no_grad(): #Deaktiviert Gradientberechnung für Testphase
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1) #Vorhergesagte Klasse ist die mit dem höchsten Score
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f'Accuracy of the model on the 10000 test images: {100 * correct / total:.2f} %')


#Visualisierung eines zufälligen Testbildes und der Vorhersage
images, labels = next(iter(test_loader)) 
random = random.randint(0, batch_size-1) 
img = images[random]
label = labels[random]

plt.imshow(img[0], cmap='gray') 
plt.title(f'Label: {label.item()}')
plt.show()

model.eval() #Setzt das Modell in den Evaluierungsmodus
while torch.no_grad():
    output = model(img.unsqueeze(0))
    _, predicted = torch.max(output.data, 1)
print(f'Das Netzwerk sagt: {predicted}')

