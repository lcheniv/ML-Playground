import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

# Set up the dataset
X = torch.tensor([[1., 2.], [2., 3.], [3., 1.], [4., 2.]])
Y = torch.tensor([1., 1., -1., -1.])

# Define the SVM model
class SVM(torch.nn.Module):
    def __init__(self):
        super(SVM, self).__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, x):
        return self.linear(x)

model = SVM()

# Set up the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Train the model
for epoch in range(1000):
    # Forward pass
    y_pred = model(X)

    # Compute loss
    hinge_loss = torch.mean(torch.clamp(1 - Y * y_pred, min=0))

    # Zero gradients, backward pass, and update weights
    optimizer.zero_grad()
    hinge_loss.backward()
    optimizer.step()

# Make predictions
predicted_classes = torch.sign(model(X)).squeeze()
print(predicted_classes)
