import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the model
class DecisionTree(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DecisionTree, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out = torch.relu(self.layer1(x))
        out = self.layer2(out)
        return out

# Define the custom loss function
def decision_loss(output, target, threshold):
    left_mask = (target <= threshold).float()
    right_mask = (target > threshold).float()
    left_error = torch.mean(torch.pow(output*left_mask - target*left_mask, 2))
    right_error = torch.mean(torch.pow(output*right_mask - target*right_mask, 2))
    total_error = left_error + right_error
    return total_error

# Generate some dummy data
x_train = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
y_train = np.array([[0.0], [0.0], [1.0], [1.0], [1.0]])

# Convert the data to PyTorch tensors
x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train).float()

# Define the model and train it
input_size = 1
hidden_size = 10
output_size = 1
model = DecisionTree(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

num_epochs = 1000
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(x_train)
    loss = decision_loss(outputs, y_train, 0.5)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print the loss every 100 epochs
    if (epoch+1) % 100 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# Test the model
x_test = torch.Tensor([[6.0]])
predicted = model(x_test).item()
print('Predicted value for input {} is: {:.4f}'.format(x_test.item(), predicted))