import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

        # First layer
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=32,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Second layer
        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=32,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2,
                                  stride=2,
                                  ceil_mode=True)

        # Third layer
        self.conv3 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        # Forth layer
        self.conv4 = nn.Conv2d(in_channels=64,
                               out_channels=64,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        # Fifth layer
        self.fc5 = nn.Linear(20480*2, 1024)

        # Output layer
        self.fc6 = nn.Linear(1024, 11)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.dropout(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = torch.flatten(x, start_dim=1, end_dim=3)

        x = self.dropout(x)
        x = self.fc5(x)
        x = self.sigmoid(x)

        x = self.fc6(x)
        x = self.softmax(x)

        return x

    def predict(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)

        x = torch.flatten(x, start_dim=1, end_dim=3)

        x = self.fc5(x)
        x = self.sigmoid(x)

        x = self.fc6(x)
        x = self.softmax(x)

        return x