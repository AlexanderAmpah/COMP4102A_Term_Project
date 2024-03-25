import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        # self.res1 = nn.Sequential(
        #     nn.Conv2d(64, 64, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 64, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True)
        # )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        # self.res2 = nn.Sequential(
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True)
        # )
        
        # Define fully connected layers
        self.fc1 = nn.Linear(256 * 7 * 7, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 26)

    def forward(self, xb):
        x = self.conv1(xb)
        x = self.conv2(x)
        #x = self.res1(x) + x
        x = self.conv3(x)
        x = self.conv4(x)
        #x = self.res2(x) + x
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load the EMNIST letters dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.EMNIST(root='./data', split='letters', train=True, download=True, transform=transform)
test_dataset = datasets.EMNIST(root='./data', split='letters', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize the model, optimizer, and loss function
model = CNN()
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# Train the model
print("Training model")
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        labels -=1
        labels.view(-1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

# Evaluate the model on the test set
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print(f"Correct: {correct}, Total {total}")
print(f"Test Accuracy: {correct / total}")

# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
#                                    nn.BatchNorm2d(32))
#         self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1),
#                                    nn.BatchNorm2d(64))
#         self.res1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1),
#                                    nn.ReLU(inplace=True),
#                                    nn.Conv2d(64,64, kernel_size=3, padding=1),
#                                    nn.ReLU(inplace=True))
#         self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1),
#                                    nn.BatchNorm2d(128))
#         self.conv4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1),
#                                    nn.BatchNorm2d(256))
#         self.res2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1),
#                                    nn.ReLU(inplace=True),
#                                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
#                                    nn.ReLU(inplace=True))
#         self.fc1 = nn.Linear(256 * 7 * 7, 1024)
#         self.fc2 = nn.Linear(1024, 256)
#         self.fc3 = nn.Linear(256, 26)

#     def forward(self, xb):
#         x = torch.relu(self.conv1(xb))
#         x = torch.max_pool2d(x, kernel_size=2, stride=2)
#         x = torch.relu(self.conv2(x))
#         x = torch.max_pool2d(x, kernel_size=2, stride=2)
#         x = self.res1(x) + x
#         x = torch.relu(self.conv3(x))
#         x = torch.max_pool2d(x, kernel_size=2, stride=2)
#         x = torch.relu(self.conv4(x))
#         x = torch.max_pool2d(x, kernel_size=2, stride=2)
#         x = self.res2(x) + x
#         x = x.view(x.size(0), -1)
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x  = self.fc3(x)
#         return x