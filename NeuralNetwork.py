import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the dataset
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.1)

# Train the neural network
net = Net()
for epoch in range(10000):
    optimizer.zero_grad()
    output = net(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

# Make predictions
predictions = net(X)
print(predictions)