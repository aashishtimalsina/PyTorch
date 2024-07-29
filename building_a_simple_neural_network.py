import torch
import torch.nn as nn
import torch.optim as optim

# Check if GPU is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a simple feedforward neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(2, 2)  # Input layer to hidden layer
        self.relu = nn.ReLU()       # Activation function
        self.fc2 = nn.Linear(2, 1)  # Hidden layer to output layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Create the network
model = SimpleNN().to(device)

# Define a loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Dummy input and target
inputs = torch.tensor([[1.0, 2.0]], device=device)
target = torch.tensor([[1.0]], device=device)

# Forward pass
output = model(inputs)
loss = criterion(output, target)

# Backward pass and optimization
optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f'Output: {output.item()}, Loss: {loss.item()}')
